import step2
import torch
import torchvision.transforms as transforms
from writing_custom_datasets import CUB_200
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import time

def create_semi_supervised_dataloader(dataset, aug1, aug2, unlabeled_ratio=0.6, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
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

def step3_trian(use_chinese=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if use_chinese:
        print(f"使用的设备: {device}")

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
        print(f"\n{'='*50}")
        print(f"Experiment with Unlabeled Ratio: {unlabeled_ratio}")
        if use_chinese:
            print(f"未标记数据比例实验: {unlabeled_ratio}")
        print(f"{'='*50}")
        train_dataset = CUB_200(root='CUB-200', train=True, download=True)

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = CUB_200(root='CUB-200', train=False, download=True, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        semi_dataloader = create_semi_supervised_dataloader(
            dataset=train_dataset,
            aug1=aug1,
            aug2=aug2,
            unlabeled_ratio=unlabeled_ratio,
            batch_size=64,
            pin_memory=True
        )

        StudentNet = step2.models.resnet18(pretrained=True)
        TeacherNet = step2.models.resnet50(pretrained=True)
        num_classes = 200
        StudentNet.fc = torch.nn.Linear(StudentNet.fc.in_features, num_classes)
        TeacherNet.fc = torch.nn.Linear(TeacherNet.fc.in_features, num_classes)

        teacher_weights_path = 'model_checkpoints/best_model.pth'
        checkpoint = torch.load(teacher_weights_path)
        TeacherNet.load_state_dict(checkpoint['model_state_dict'])
        TeacherNet.eval()
        TeacherNet.to(device)  # 确保 TeacherNet 在正确的设备上
        print(f"Loaded TeacherNet weights from '{teacher_weights_path}'.")
        if use_chinese:
            print(f"从 '{teacher_weights_path}' 加载 TeacherNet 权重。")

        def validate_teacher(model, dataloader, device, use_chinese=False):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in dataloader:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images.to(device))  # 确保模型和输入都在同一设备上
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy

        teacher_accuracy = validate_teacher(TeacherNet, val_dataloader, device, use_chinese)
        print(f"TeacherNet Validation Accuracy: {teacher_accuracy:.2f}%")
        if use_chinese:
            print(f"TeacherNet 验证集准确率: {teacher_accuracy:.2f}%")

        print("StudentNet and TeacherNet initialized.")
        if use_chinese:
            print("学生网络和教师网络已初始化。")

        for param in TeacherNet.parameters():
            param.requires_grad = False
        print("TeacherNet parameters frozen.")
        if use_chinese:
            print("教师网络参数已冻结。")

        classification_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none').to(device)
        optimizer = torch.optim.Adam(StudentNet.parameters(), lr=0.001)

        StudentNet.to(device)

        def calculate_confidence(teacher_output_aug1, teacher_output_aug2):
            prob_aug1 = F.softmax(teacher_output_aug1, dim=-1)
            prob_aug2 = F.softmax(teacher_output_aug2, dim=-1)
            l1_distance = torch.sum(torch.abs(prob_aug1 - prob_aug2), dim=-1)
            confidence = 1.0 - l1_distance / 2.0
            return confidence

        def train_one_epoch(epoch, semi_dataloader, StudentNet, TeacherNet, optimizer, classification_criterion, device, use_chinese=False):
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
                    if use_chinese:
                        pass  # 简化中文输出
                    with torch.no_grad():
                        teacher_output_aug1 = TeacherNet(aug1_images[unlabeled_mask].to(device)) # 确保输入在正确的设备上
                        teacher_output_aug2 = TeacherNet(aug2_images[unlabeled_mask].to(device)) # 确保输入在正确的设备上
                        predicted_labels_aug1 = torch.argmax(teacher_output_aug1, dim=-1)
                        pseudo_labels[unlabeled_mask] = predicted_labels_aug1

                        confidence_scores = calculate_confidence(teacher_output_aug1, teacher_output_aug2)
                        confidence_weights[unlabeled_mask] = confidence_scores

                student_predictions = StudentNet(aug1_images)

                labeled_mask_tensor = is_labeled.to(device)
                labeled_loss = (classification_criterion(student_predictions[labeled_mask_tensor], labels[labeled_mask_tensor])).mean() if labeled_mask_tensor.any() else torch.tensor(0.0).to(device)
                running_labeled_loss += labeled_loss.item()

                weighted_unlabeled_loss = torch.tensor(0.0).to(device)
                if unlabeled_mask.any():
                    unlabeled_predictions = student_predictions[unlabeled_mask]
                    unlabeled_loss_values = classification_criterion(unlabeled_predictions, pseudo_labels[unlabeled_mask])
                    weighted_unlabeled_loss = (unlabeled_loss_values * confidence_weights[unlabeled_mask]).mean()
                running_unlabeled_loss += weighted_unlabeled_loss.item()

                loss = labeled_loss + weighted_unlabeled_loss
                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    if use_chinese:
                        print(f"Epoch [{epoch+1}], Batch [{i+1}/{total_batches}], 标记损失: {labeled_loss.item():.4f}, 未标记损失: {weighted_unlabeled_loss.item():.4f}")
                    else:
                        print(f"Epoch [{epoch+1}], Batch [{i+1}/{total_batches}], Labeled Loss: {labeled_loss.item():.4f}, Unlabeled Loss: {weighted_unlabeled_loss.item():.4f}")

            epoch_duration = time.time() - start_time
            avg_labeled_loss = running_labeled_loss / total_batches
            avg_unlabeled_loss = running_unlabeled_loss / total_batches
            if use_chinese:
                print(f"Epoch [{epoch+1}] 完成，耗时 {epoch_duration:.2f} 秒, 平均标记损失: {avg_labeled_loss:.4f}, 平均未标记损失: {avg_unlabeled_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}] finished in {epoch_duration:.2f} seconds, Avg Labeled Loss: {avg_labeled_loss:.4f}, Avg Unlabeled Loss: {avg_unlabeled_loss:.4f}")

        def validate(model, dataloader, device, use_chinese=False):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in dataloader:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
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
        if use_chinese:
            print("开始训练...")
        for epoch in range(num_epochs):
            train_one_epoch(epoch, semi_dataloader, StudentNet, TeacherNet, optimizer, classification_criterion, device, use_chinese)
            current_val_accuracy = validate(StudentNet, val_dataloader, device, use_chinese)

            if current_val_accuracy > best_val_accuracy * 1.01:
                best_val_accuracy = current_val_accuracy
                best_model_wts = copy.deepcopy(StudentNet.state_dict())
                epochs_without_improvement = 0
                if use_chinese:
                    print(f"验证精度提升至 {best_val_accuracy:.2f}%, 正在保存模型。")
                else:
                    print(f"Validation accuracy improved to {best_val_accuracy:.2f}%, saving model.")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    if use_chinese:
                        print("触发早停。")
                    else:
                        print("Early stopping triggered.")
                    break

        print("Training finished for unlabeled ratio:", unlabeled_ratio)
        if use_chinese:
            print("针对未标记数据比例的训练已完成:", unlabeled_ratio)
        StudentNet.load_state_dict(best_model_wts)
        torch.save(StudentNet.state_dict(), f'student_net_unlabeled_{unlabeled_ratio:.1f}.pth')
        if use_chinese:
            print(f"最佳模型已保存至 student_net_unlabeled_{unlabeled_ratio:.1f}.pth")
        else:
            print(f"Best model saved to student_net_unlabeled_{unlabeled_ratio:.1f}.pth")
        all_results[unlabeled_ratio] = best_val_accuracy

    print(f"\n{'='*50}")
    if use_chinese:
        print("不同未标记比例的最终结果:")
        for ratio, accuracy in all_results.items():
            print(f"未标记比例: {ratio:.1f}, 最佳验证精度: {accuracy:.2f}%")
    else:
        print("Final Results for Different Unlabeled Ratios:")
        for ratio, accuracy in all_results.items():
            print(f"Unlabeled Ratio: {ratio:.1f}, Best Validation Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    step3_trian(use_chinese=True)