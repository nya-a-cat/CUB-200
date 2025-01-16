# step3.py
import step2
import torch
import torchvision.transforms as transforms
from writing_custom_datasets import CUB_200
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
            - 'label': Original label (or None if unlabeled)
            - 'is_labeled': Boolean indicating if the sample is labeled
    """

    class SemiSupervisedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, augmentation1, augmentation2, unlabeled_ratio):
            self.base_dataset = base_dataset
            self.aug1 = augmentation1
            self.aug2 = augmentation2
            self.unlabeled_ratio = unlabeled_ratio
            self.labeled_indices = []
            self.unlabeled_indices = []
            self._prepare_data()

        def _prepare_data(self):
            images_by_class = {}
            for idx in range(len(self.base_dataset)):
                img, label_int = self.base_dataset[idx]  # Access data using [] indexing
                if label_int not in images_by_class:
                    images_by_class[label_int] = []
                images_by_class[label_int].append((self.base_dataset.filenames[idx], idx))

            for label_int, items in images_by_class.items():
                sorted_items = sorted(items, key=lambda x: x[0])
                num_samples = len(sorted_items)
                num_unlabeled = int(unlabeled_ratio * num_samples)
                for i, (filename, original_index) in enumerate(sorted_items):
                    if i < num_unlabeled:
                        self.unlabeled_indices.append(original_index)
                    else:
                        self.labeled_indices.append(original_index)

            print(f"Number of labeled samples: {len(self.labeled_indices)}")
            print(f"Number of unlabeled samples: {len(self.unlabeled_indices)}")
            self.all_indices = self.labeled_indices + self.unlabeled_indices

        def __getitem__(self, idx):
            original_index = self.all_indices[idx]
            image, label = self.base_dataset[original_index]
            is_labeled = original_index in self.labeled_indices
            return {
                'aug1': self.aug1(image),
                'aug2': self.aug2(image),
                'label': label if is_labeled else -1,  # Use a placeholder for unlabeled
                'is_labeled': is_labeled
            }

        def __len__(self):
            return len(self.all_indices)

    semi_dataset = SemiSupervisedDataset(dataset, aug1, aug2, unlabeled_ratio)
    return DataLoader(
        semi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

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

    R_values = [0.4, 0.6, 0.8]

    for unlabeled_ratio in R_values:
        print(f"\nExperiment with Unlabeled Ratio: {unlabeled_ratio}")
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

        for param in TeacherNet.parameters():
            param.requires_grad = False

        classification_criterion = torch.nn.CrossEntropyLoss(reduction='none') # Keep individual losses

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
            fig.suptitle(f'Pseudo-Labels and Confidence (Unlabeled Ratio: {unlabeled_ratio})', fontsize=16)

            def tensor_to_img(tensor):
                img = tensor.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return img

            for i in range(num_samples):
                axes[i, 0].imshow(tensor_to_img(aug1_images[i]))
                title = "Labeled" if is_labeled[i] else f"Unlabeled\nPseudo-label: {torch.argmax(teacher_preds_aug1[i]).item()}"
                axes[i, 0].set_title(title)
                axes[i, 0].axis('off')

                axes[i, 1].imshow(tensor_to_img(aug2_images[i]))
                title = "Labeled" if is_labeled[i] else f"Unlabeled\nPseudo-label: {torch.argmax(teacher_preds_aug2[i]).item()}"
                axes[i, 1].set_title(title)
                axes[i, 1].axis('off')

                if not is_labeled[i]:
                    axes[i, 2].text(0.5, 0.5, f"Confidence: {confidence_scores[i].item():.4f}", ha='center', va='center')
                axes[i, 2].axis('off')

            plt.tight_layout()
            plt.show()

        # Example usage of the semi-supervised dataloader and pseudo-labeling
        for i, batch in enumerate(semi_dataloader):
            aug1_images = batch['aug1']
            aug2_images = batch['aug2']
            labels = batch['label']
            is_labeled = batch['is_labeled']

            # 3.1) 当data_loader吐出的样本无标签的时候，使用TeacherNet预测的类别当作伪标签
            pseudo_labels = torch.zeros_like(labels)
            confidence_weights = torch.ones_like(labels, dtype=torch.float)

            unlabeled_mask = ~is_labeled
            if unlabeled_mask.any():
                with torch.no_grad():
                    teacher_output_aug1 = TeacherNet(aug1_images[unlabeled_mask])
                    teacher_output_aug2 = TeacherNet(aug2_images[unlabeled_mask])
                    pseudo_labels[unlabeled_mask] = torch.argmax(teacher_output_aug1, dim=-1)

                    # 3.2) 使用TeacherNet在aug1(x)和aug2(x)上预测结果的差异来衡量伪标签的置信度
                    confidence_scores = calculate_confidence(teacher_output_aug1, teacher_output_aug2)
                    confidence_weights[unlabeled_mask] = confidence_scores

                # 3.3) 设计可视化来显示伪标签及其置信度. Only visualize if there are unlabeled samples
                visualize_pseudo_labels(
                    batch,
                    teacher_output_aug1,
                    teacher_output_aug2,
                    confidence_scores
                )
                break # Visualize only the first batch for brevity

            # You would typically use pseudo_labels and confidence_weights in your training loop
            # For example, when calculating the classification loss for unlabeled data:
            # if unlabeled_mask.any():
            #     unlabeled_predictions = StudentNet(aug1_images[unlabeled_mask])
            #     unlabeled_loss = classification_criterion(unlabeled_predictions, pseudo_labels[unlabeled_mask])
            #     weighted_unlabeled_loss = (unlabeled_loss * confidence_weights[unlabeled_mask]).mean()

            if i > 2:  # Process only a few batches for demonstration
                break