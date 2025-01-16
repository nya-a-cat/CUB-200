# step3.py
import step2
import torch
import torchvision.transforms as transforms
from writing_custom_datasets import CUB_200  # Import the CUB_200 dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 30})  # 可以尝试 12, 14, 16 或更大的值
# 全局设置标题字体大小



def create_semi_supervised_dataloader(dataset, aug1, aug2, unlabeled_ratio=0.6, batch_size=32, shuffle=True, num_workers=0):
    """
    Creates a DataLoader with a portion of unlabeled data.

    Args:
        dataset: Base dataset (e.g., CUB_200)
        aug1: First augmentation transform
        aug2: Second augmentation transform
        unlabeled_ratio: The ratio of unlabeled data (0 to 1)
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader that yields dictionaries containing:
            - 'aug1': First augmented view
            - 'aug2': Second augmented view
            - 'label': Original label (or -1 if unlabeled)
            - 'is_labeled': Boolean indicating if the sample is labeled
    """
    print("Creating semi-supervised DataLoader...")

    class SemiSupervisedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, augmentation1, augmentation2, unlabeled_ratio):
            print("Initializing SemiSupervisedDataset...")
            self.base_dataset = base_dataset
            self.aug1 = augmentation1
            self.aug2 = augmentation2
            self.unlabeled_ratio = unlabeled_ratio
            self.labeled_indices = []
            self.unlabeled_indices = []
            self._prepare_data()
            print("SemiSupervisedDataset initialized.")

        def _prepare_data(self):
            print("Preparing semi-supervised data...")
            images_by_class = {}
            for idx in range(len(self.base_dataset)):
                image, label_int = self.base_dataset[idx]
                if label_int not in images_by_class:
                    images_by_class[label_int] = []
                # Store filename and original index
                filename = self.base_dataset.filtered_info.iloc[idx]['filepath']
                images_by_class[label_int].append((filename.split('/')[-1], idx))

            for label_int, items in images_by_class.items():
                # Sort by filename
                sorted_items = sorted(items, key=lambda x: x[0])
                num_samples = len(sorted_items)
                num_unlabeled = int(self.unlabeled_ratio * num_samples)
                print(f"Class {label_int}: Total {num_samples} images, marking first {num_unlabeled} as unlabeled.")
                for i, (filename, original_index) in enumerate(sorted_items):
                    if i < num_unlabeled:
                        self.unlabeled_indices.append(original_index)
                    else:
                        self.labeled_indices.append(original_index)

            print(f"Number of labeled samples: {len(self.labeled_indices)}")
            print(f"Number of unlabeled samples: {len(self.unlabeled_indices)}")
            self.all_indices = self.labeled_indices + self.unlabeled_indices
            print("Semi-supervised data preparation complete.")

        def __getitem__(self, idx):
            original_index = self.all_indices[idx]
            image, label = self.base_dataset[original_index]
            is_labeled = original_index in self.labeled_indices
            return {
                'aug1': self.aug1(image),
                'aug2': self.aug2(image),
                'label': label if is_labeled else -1,  # Use -1 as placeholder for unlabeled
                'is_labeled': is_labeled
            }

        def __len__(self):
            return len(self.all_indices)

    semi_dataset = SemiSupervisedDataset(dataset, aug1, aug2, unlabeled_ratio)
    print("Creating DataLoader instance...")
    dataloader = DataLoader(
        semi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    print("DataLoader created.")
    return dataloader

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    aug1 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    aug2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3.4) 分别取 R=0.4, 0.6, 0.8 进行实验
    R_values = [0.4, 0.6, 0.8]

    for unlabeled_ratio in R_values:
        print(f"\n{'='*40}")
        print(f"Experiment with Unlabeled Ratio: {unlabeled_ratio}")
        print(f"{'='*40}")
        train_dataset = CUB_200(root='CUB-200', train=True, download=True)
        semi_dataloader = create_semi_supervised_dataloader(
            dataset=train_dataset,
            aug1=aug1,
            aug2=aug2,
            unlabeled_ratio=unlabeled_ratio,
            batch_size=4  # 减小 batch size 以方便可视化
        )

        StudentNet = step2.models.resnet18(pretrained=True)
        TeacherNet = step2.models.resnet50(pretrained=True)
        num_classes = 200
        StudentNet.fc = torch.nn.Linear(StudentNet.fc.in_features, num_classes)
        TeacherNet.fc = torch.nn.Linear(TeacherNet.fc.in_features, num_classes)
        TeacherNet.eval()  # Set TeacherNet to eval mode
        print("StudentNet and TeacherNet initialized.")

        for param in TeacherNet.parameters():
            param.requires_grad = False
        print("TeacherNet parameters frozen.")

        classification_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none') # Keep individual losses

        def calculate_confidence(teacher_output_aug1, teacher_output_aug2):
            """Calculates confidence based on the difference between teacher's predictions."""
            prob_aug1 = torch.softmax(teacher_output_aug1, dim=-1)
            prob_aug2 = torch.softmax(teacher_output_aug2, dim=-1)
            # Using negative absolute difference of probabilities as confidence
            confidence = 1.0 - torch.abs(prob_aug1 - prob_aug2).sum(dim=-1)
            return confidence

        def visualize_pseudo_labels(batch, teacher_preds_aug1, teacher_preds_aug2, confidence_scores):
            aug1_images = batch['aug1']
            aug2_images = batch['aug2']
            is_labeled = batch['is_labeled'].cpu().numpy()
            num_samples = aug1_images.size(0)

            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
            fig.suptitle(f'Pseudo-Labels and Confidence (Unlabeled Ratio: {unlabeled_ratio:.2f})', fontsize=16)

            def tensor_to_img(tensor):
                img = tensor.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return img

            for i in range(num_samples):
                axes[i, 0].imshow(tensor_to_img(aug1_images[i]))
                title = "Labeled" if is_labeled[i] else f"Unlabeled\nPseudo-label (aug1): {torch.argmax(teacher_preds_aug1[i]).item()}" if teacher_preds_aug1 is not None and teacher_preds_aug1.size(0) > i else "Unlabeled"
                axes[i, 0].set_title(title)
                axes[i, 0].axis('off')

                axes[i, 1].imshow(tensor_to_img(aug2_images[i]))
                title = "Labeled" if is_labeled[i] else f"Unlabeled\nPseudo-label (aug2): {torch.argmax(teacher_preds_aug2[i]).item()}" if teacher_preds_aug2 is not None and teacher_preds_aug2.size(0) > i else "Unlabeled"
                axes[i, 1].set_title(title)
                axes[i, 1].axis('off')

                if not is_labeled[i]:
                    conf_text = f"Confidence: {confidence_scores[i].item():.4f}" if confidence_scores is not None and confidence_scores.size(0) > i else ""
                    axes[i, 2].text(0.5, 0.5, conf_text, ha='center', va='center')
                else:
                    axes[i, 2].text(0.5, 0.5, "Labeled Data", ha='center', va='center')
                axes[i, 2].axis('off')

            plt.tight_layout()
            plt.show()

        print("Starting iteration over DataLoader...")
        # Example usage of the semi-supervised dataloader and pseudo-labeling
        for i, batch in enumerate(semi_dataloader):
            print(f"\n--- Batch {i+1} ---")
            aug1_images = batch['aug1']
            aug2_images = batch['aug2']
            labels = batch['label']
            is_labeled = batch['is_labeled']

            print("Batch loaded.")
            # 3.1) 当data_loader吐出的样本无标签的时候，使用TeacherNet预测的类别当作伪标签
            pseudo_labels = torch.zeros_like(labels)
            confidence_weights = torch.ones_like(labels, dtype=torch.float)

            teacher_output_aug1 = None
            teacher_output_aug2 = None
            confidence_scores = None

            unlabeled_mask = ~is_labeled
            if unlabeled_mask.any():
                print("Unlabeled data found in the batch. Generating pseudo-labels...")
                with torch.no_grad():
                    teacher_output_aug1 = TeacherNet(aug1_images[unlabeled_mask])
                    teacher_output_aug2 = TeacherNet(aug2_images[unlabeled_mask])
                    predicted_labels_aug1 = torch.argmax(teacher_output_aug1, dim=-1)
                    predicted_labels_aug2 = torch.argmax(teacher_output_aug2, dim=-1)
                    pseudo_labels[unlabeled_mask] = predicted_labels_aug1 # Using predictions from aug1

                    # 3.2) 使用TeacherNet在aug1(x)和aug2(x)上预测结果的差异来衡量伪标签的置信度
                    confidence_scores = calculate_confidence(teacher_output_aug1, teacher_output_aug2)
                    confidence_weights[unlabeled_mask] = confidence_scores
                print("Pseudo-labels and confidence scores generated.")

            # 3.3) 设计可视化来显示伪标签及其置信度. Only visualize if there are unlabeled samples
            print("Visualizing pseudo-labels and confidence...")
            visualize_pseudo_labels(
                batch,
                teacher_output_aug1,
                teacher_output_aug2,
                confidence_scores
            )
            print("Visualization complete.")

            # You would typically use pseudo_labels and confidence_weights in your training loop
            if unlabeled_mask.any():
                print("Calculating weighted unlabeled loss (example)...")
                unlabeled_predictions = StudentNet(aug1_images[unlabeled_mask])
                # Use pseudo_labels and confidence_weights for loss calculation
                unlabeled_loss = classification_criterion(unlabeled_predictions, pseudo_labels[unlabeled_mask])
                weighted_unlabeled_loss = (unlabeled_loss * confidence_weights[unlabeled_mask]).mean()
                print(f"Weighted Unlabeled Loss: {weighted_unlabeled_loss.item()}")

            if i > 2:  # Process only a few batches for demonstration
                print("Stopping after a few batches for demonstration.")
                break
        print("Finished iteration over DataLoader.")