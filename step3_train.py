# step3_test.py
import step2
import torch
import torchvision.transforms as transforms
from writing_custom_datasets import CUB_200
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import time

def create_semi_supervised_dataloader(dataset, aug1, aug2, unlabeled_ratio=0.6, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
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
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader that yields dictionaries containing:
            - 'aug1': First augmented view
            - 'aug2': Second augmented view
            - 'label': Original label (or -1 if unlabeled)
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
                image, label_int = self.base_dataset[idx]
                if label_int not in images_by_class:
                    images_by_class[label_int] = []
                filename = self.base_dataset.filtered_info.iloc[idx]['filepath']
                images_by_class[label_int].append((filename.split('/')[-1], idx))

            for label_int, items in images_by_class.items():
                sorted_items = sorted(items, key=lambda x: x[0])
                num_samples = len(sorted_items)
                num_unlabeled = int(self.unlabeled_ratio * num_samples)
                for i, (filename, original_index) in enumerate(sorted_items):
                    if i < num_unlabeled:
                        self.unlabeled_indices.append(original_index)
                    else:
                        self.labeled_indices.append(original_index)

            self.all_indices = self.labeled_indices + self.unlabeled_indices

        def __getitem__(self, idx):
            original_index = self.all_indices[idx]
            image, label = self.base_dataset[original_index]
            is_labeled = original_index in self.labeled_indices
            return {
                'aug1': self.aug1(image),
                'aug2': self.aug2(image),
                'label': label if is_labeled else -1,
                'is_labeled': is_labeled
            }

        def __len__(self):
            return len(self.all_indices)

    semi_dataset = SemiSupervisedDataset(dataset, aug1, aug2, unlabeled_ratio)
    dataloader = DataLoader(
        semi_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader

def test_step3():
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    all_results = {}

    for unlabeled_ratio in R_values:
        print(f"\n{'='*40}")
        print(f"Experiment with Unlabeled Ratio: {unlabeled_ratio}")
        print(f"{'='*40}")
        train_dataset = CUB_200(root='CUB-200', train=True, download=True)

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = CUB_200(root='CUB-200', train=False, download=True, transform=val_transform)

        semi_dataloader = create_semi_supervised_dataloader(
            dataset=train_dataset,
            aug1=aug1,
            aug2=aug2,
            unlabeled_ratio=unlabeled_ratio,
            batch_size=64,
            pin_memory=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        StudentNet = step2.models.resnet18(pretrained=True)
        TeacherNet = step2.models.resnet50(pretrained=True)
        num_classes = 200
        StudentNet.fc = torch.nn.Linear(StudentNet.fc.in_features, num_classes)
        TeacherNet.fc = torch.nn.Linear(TeacherNet.fc.in_features, num_classes)
        TeacherNet.eval()
        print("StudentNet and TeacherNet initialized.")

        for param in TeacherNet.parameters():
            param.requires_grad = False
        print("TeacherNet parameters frozen.")

        classification_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(device)
        optimizer = torch.optim.Adam(StudentNet.parameters(), lr=0.001)

        StudentNet.to(device)
        TeacherNet.to(device)

        def calculate_confidence(teacher_output_aug1, teacher_output_aug2):
            prob_aug1 = torch.softmax(teacher_output_aug1, dim=-1)
            prob_aug2 = torch.softmax(teacher_output_aug2, dim=-1)
            confidence = 1.0 - torch.abs(prob_aug1 - prob_aug2).sum(dim=-1)
            return confidence

        def train_one_epoch(epoch, semi_dataloader, StudentNet, TeacherNet, optimizer, classification_criterion, device):
            StudentNet.train()
            running_labeled_loss = 0.0
            running_unlabeled_loss = 0.0
            total_batches = len(semi_dataloader)
            start_time = time.time()

            for i, batch in enumerate(semi_dataloader):
                aug1_images = batch['aug1'].to(device)
                aug2_images = batch['aug2'].to(device)
                labels = batch['label'].to(device)
                is_labeled = batch['is_labeled']

                optimizer.zero_grad()

                pseudo_labels = torch.zeros_like(labels)
                confidence_weights = torch.ones_like(labels, dtype=torch.float).to(device)

                unlabeled_mask = ~is_labeled
                if unlabeled_mask.any():
                    print(f"  Unlabeled Data in Batch:")
                    print(f"    Number of Unlabeled Samples: {unlabeled_mask.sum()}")
                    with torch.no_grad():
                        teacher_output_aug1 = TeacherNet(aug1_images[unlabeled_mask])
                        teacher_output_aug2 = TeacherNet(aug2_images[unlabeled_mask])
                        predicted_labels_aug1 = torch.argmax(teacher_output_aug1, dim=-1)
                        pseudo_labels[unlabeled_mask] = predicted_labels_aug1

                        confidence_scores = calculate_confidence(teacher_output_aug1, teacher_output_aug2)
                        confidence_weights[unlabeled_mask] = confidence_scores
                        print(f"    Example Pseudo-Labels: {predicted_labels_aug1[:5].tolist()}")
                        print(f"    Example Confidence Scores: {confidence_scores[:5].tolist()}")
                        print(f"    Shape of Confidence Weights: {confidence_weights[unlabeled_mask].shape}")

                student_predictions = StudentNet(aug1_images)

                labeled_mask_tensor = is_labeled.to(device)
                labeled_loss = (classification_criterion(student_predictions[labeled_mask_tensor], labels[labeled_mask_tensor])).mean() if labeled_mask_tensor.any() else torch.tensor(0.0).to(device)
                running_labeled_loss += labeled_loss.item()

                weighted_unlabeled_loss = torch.tensor(0.0).to(device)
                if unlabeled_mask.any():
                    unlabeled_predictions = student_predictions[unlabeled_mask]
                    unlabeled_loss_values = classification_criterion(unlabeled_predictions, pseudo_labels[unlabeled_mask])
                    weighted_unlabeled_loss = (unlabeled_loss_values * confidence_weights[unlabeled_mask]).mean()
                    print(f"    Weighted Unlabeled Loss: {weighted_unlabeled_loss.item():.4f}")
                running_unlabeled_loss += weighted_unlabeled_loss.item()

                loss = labeled_loss + weighted_unlabeled_loss
                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}], Batch [{i+1}/{total_batches}], Labeled Loss: {labeled_loss.item():.4f}, Unlabeled Loss: {weighted_unlabeled_loss.item():.4f}")

            epoch_duration = time.time() - start_time
            avg_labeled_loss = running_labeled_loss / total_batches
            avg_unlabeled_loss = running_unlabeled_loss / total_batches
            print(f"Epoch [{epoch+1}] finished in {epoch_duration:.2f} seconds, Avg Labeled Loss: {avg_labeled_loss:.4f}, Avg Unlabeled Loss: {avg_unlabeled_loss:.4f}")

        def validate():
            StudentNet.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = StudentNet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy

        num_epochs = 30
        best_val_accuracy = 0.0
        best_model_wts = copy.deepcopy(StudentNet.state_dict())
        epochs_without_improvement = 0
        patience = 5

        print("Starting training...")
        for epoch in range(num_epochs):
            train_one_epoch(epoch, semi_dataloader, StudentNet, TeacherNet, optimizer, classification_criterion, device)
            current_val_accuracy = validate()

            if current_val_accuracy > best_val_accuracy * 1.01:
                best_val_accuracy = current_val_accuracy
                best_model_wts = copy.deepcopy(StudentNet.state_dict())
                epochs_without_improvement = 0
                print(f"Validation accuracy improved to {best_val_accuracy:.2f}%, saving model.")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training finished for unlabeled ratio:", unlabeled_ratio)
        StudentNet.load_state_dict(best_model_wts)
        torch.save(StudentNet.state_dict(), f'student_net_unlabeled_{unlabeled_ratio:.1f}.pth')
        print(f"Best model saved to student_net_unlabeled_{unlabeled_ratio:.1f}.pth")
        all_results[unlabeled_ratio] = best_val_accuracy

    print("\n"+"="*40)
    print("Final Results for Different Unlabeled Ratios:")
    for ratio, accuracy in all_results.items():
        print(f"Unlabeled Ratio: {ratio:.1f}, Best Validation Accuracy: {accuracy:.2f}%")
    print("="*40)

if __name__ == '__main__':
    test_step3()