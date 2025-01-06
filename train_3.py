import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from typing import Optional, Tuple
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class SemiSupervisedCUB200(Dataset):
    """
    实现3要求：将数据集分为有标签和无标签两部分
    R: 无标签数据的比例 (40%, 60%, or 80%)
    """

    def __init__(self,
                 root: str,
                 unlabel_ratio: float = 0.6,  # R=60 by default
                 transform=None,
                 train: bool = True):
        self.root = root
        self.transform = transform
        self.train = train
        self.unlabel_ratio = unlabel_ratio

        # 加载原始数据集
        self.data = []
        self.targets = []
        self.has_label = []  # 标记每个样本是否有标签

        # 按类别整理数据
        class_data = defaultdict(list)
        for img_path, label in self._load_data():
            class_data[label].append(img_path)

        # 对每个类别的图片按文件名排序，并按比例删除标签
        for label, images in class_data.items():
            sorted_images = sorted(images)
            unlabel_count = int(len(sorted_images) * unlabel_ratio)

            # 前R%作为无标签数据
            for img_path in sorted_images[:unlabel_count]:
                self.data.append(img_path)
                self.targets.append(-1)  # -1表示无标签
                self.has_label.append(False)

            # 剩余的保持原有标签
            for img_path in sorted_images[unlabel_count:]:
                self.data.append(img_path)
                self.targets.append(label)
                self.has_label.append(True)

    def _load_data(self):
        """从CUB-200数据集加载数据"""
        # 实现CUB-200数据集的加载逻辑
        pass

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, bool]:
        """返回图像、标签（如果有）和是否有标签的标志"""
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx], self.has_label[idx]

    def __len__(self):
        return len(self.data)


class SemiSupervisedDistillation:
    """
    实现3.1-3.2要求：使用教师网络生成伪标签并计算置信度
    """

    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 temperature: float = 2.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature

    def generate_pseudo_label(self,
                              aug1: torch.Tensor,
                              aug2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3.1: 使用TeacherNet生成伪标签
        3.2: 计算两个增强版本预测的一致性作为置信度
        """
        self.teacher.eval()
        with torch.no_grad():
            # 对两个增强版本进行预测
            logits1 = self.teacher(aug1)
            logits2 = self.teacher(aug2)

            # 软化logits
            soft1 = F.softmax(logits1 / self.temperature, dim=1)
            soft2 = F.softmax(logits2 / self.temperature, dim=1)

            # 计算预测一致性作为置信度
            confidence = torch.sum(soft1 * soft2, dim=1)

            # 获取伪标签（使用aug1的预测）
            pseudo_labels = torch.argmax(logits1, dim=1)

        return pseudo_labels, confidence


def train_semi_supervised(
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        visualize_freq: int = 100):
    """
    实现半监督训练循环
    """
    teacher_model.eval()
    student_model.train()

    distiller = SemiSupervisedDistillation(teacher_model, student_model)
    criterion = nn.CrossEntropyLoss(reduction='none')

    total_loss = 0
    labeled_correct = 0
    labeled_total = 0
    unlabeled_correct = 0
    unlabeled_total = 0

    for batch_idx, (aug1, aug2, labels, is_labeled) in enumerate(train_loader):
        aug1, aug2 = aug1.to(device), aug2.to(device)
        labels = labels.to(device)

        # 处理有标签数据
        labeled_mask = is_labeled
        if labeled_mask.any():
            labeled_output = student_model(aug1[labeled_mask])
            labeled_loss = criterion(labeled_output, labels[labeled_mask]).mean()
        else:
            labeled_loss = 0

        # 处理无标签数据
        unlabeled_mask = ~is_labeled
        if unlabeled_mask.any():
            # 3.1: 生成伪标签
            # 3.2: 计算置信度
            pseudo_labels, confidence = distiller.generate_pseudo_label(
                aug1[unlabeled_mask],
                aug2[unlabeled_mask]
            )

            unlabeled_output = student_model(aug1[unlabeled_mask])
            unlabeled_loss = (criterion(unlabeled_output, pseudo_labels) * confidence).mean()
        else:
            unlabeled_loss = 0

        # 总损失
        loss = labeled_loss + 0.5 * unlabeled_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录指标
        total_loss += loss.item()

        # 3.3: 定期可视化伪标签和置信度
        if batch_idx % visualize_freq == 0:
            visualize_pseudo_labels(
                pseudo_labels if unlabeled_mask.any() else None,
                confidence if unlabeled_mask.any() else None,
                batch_idx,
                epoch
            )

        # 记录到wandb
        wandb.log({
            "epoch": epoch,
            "batch": batch_idx,
            "total_loss": loss.item(),
            "labeled_loss": labeled_loss.item() if isinstance(labeled_loss, torch.Tensor) else labeled_loss,
            "unlabeled_loss": unlabeled_loss.item() if isinstance(unlabeled_loss, torch.Tensor) else unlabeled_loss,
        })

    return total_loss / len(train_loader)


def visualize_pseudo_labels(pseudo_labels: Optional[torch.Tensor],
                            confidence: Optional[torch.Tensor],
                            batch_idx: int,
                            epoch: int):
    """
    3.3: 可视化伪标签及其置信度
    """
    if pseudo_labels is None or confidence is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 伪标签分布
    if pseudo_labels is not None:
        sns.histplot(pseudo_labels.cpu().numpy(), ax=ax1)
        ax1.set_title('Pseudo Label Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')

    # 置信度分布
    if confidence is not None:
        sns.histplot(confidence.cpu().numpy(), ax=ax2)
        ax2.set_title('Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')

    plt.tight_layout()
    wandb.log({
        "pseudo_labels": wandb.Image(fig),
        "epoch": epoch,
        "batch": batch_idx
    })
    plt.close()


def main():
    """
    3.4: 实验不同的R值 (40%, 60%, 80%)
    """
    # 基础设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unlabel_ratios = [0.4, 0.6, 0.8]  # R = 40, 60, 80

    for R in unlabel_ratios:
        print(f"\nStarting experiment with {R * 100}% unlabeled data")

        wandb.init(
            project="semi-supervised-distillation",
            name=f"unlabel-ratio-{int(R * 100)}",
            config={
                "unlabel_ratio": R,
                "architecture": "resnet18_student",
                "dataset": "CUB200",
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 1e-4,
            }
        )

        # 创建半监督数据集
        train_data = SemiSupervisedCUB200(
            root='CUB-200',
            unlabel_ratio=R,
            transform=get_transforms(train=True)
        )

        test_data = CUB_200(
            root='CUB-200',
            transform=get_transforms(train=False),
            train=False
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2
        )

        test_loader = DataLoader(
            test_data,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2
        )

        # 初始化模型
        teacher_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 200)
        # 加载之前训练好的教师模型
        teacher_model.load_state_dict(torch.load('model_checkpoints/best_model.pth')['model_state_dict'])
        teacher_model = teacher_model.to(device)

        student_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)
        student_model.fc = nn.Linear(student_model.fc.in_features, 200)
        student_model = student_model.to(device)

        # 冻结教师模型参数
        for param in teacher_model.parameters():
            param.requires_grad = False

        # 优化器设置
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        # 训练循环
        best_acc = 0
        patience = 15
        patience_counter = 0

        for epoch in range(100):
            # 训练阶段
            train_loss = train_semi_supervised(
                teacher_model=teacher_model,
                student_model=student_model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                visualize_freq=100
            )

            # 验证阶段
            student_model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = student_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            # 记录到wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_accuracy": accuracy,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            # 打印进度
            print(f'Epoch {epoch}: Loss={train_loss:.4f}, Accuracy={accuracy:.2f}%')

            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                }, f'model_checkpoints/best_student_R{int(R * 100)}.pth')
                print(f'Saved new best model with accuracy: {accuracy:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1

            # 学习率调整
            scheduler.step()

            # 早停
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # 打印最终结果
        print(f"\nExperiment with {R * 100}% unlabeled data completed")
        print(f"Best accuracy: {best_acc:.2f}%")

        wandb.finish()


if __name__ == "__main__":
    main()