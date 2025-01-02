# 在 src/utils/train.py 中修改导入语句
import os
import sys

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import src.models.resnet50v2 as resnet50
from src.data.writing_custom_datasets import CUB_200
from torch.utils.data import DataLoader
from grams import Grams
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn.functional as F


class TrainingVisualizer:
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1]),
        ])

    def visualize_batch(self, images, targets, predictions=None, phase="train"):
        """可视化一个batch的图像"""
        # 反归一化图像
        images_cpu = self.denormalize(images.cpu())

        # 创建图像网格
        grid = vutils.make_grid(images_cpu[:16], nrow=4, padding=2, normalize=True)

        # 记录到wandb
        caption = f"{phase.capitalize()} Batch Images"
        if predictions is not None:
            caption += f" (Pred: {predictions[:16].tolist()})"
        wandb.log({
            f"{phase}_batch": wandb.Image(grid, caption=caption)
        })

    def visualize_feature_maps(self, images, layer_name="layer4"):
        """可视化特征图"""
        self.model.eval()
        images = images.to(self.device)

        # 获取特定层的特征图
        features = {}

        def hook_fn(module, input, output):
            features[layer_name] = output

        # 注册钩子
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(images)

        # 移除钩子
        handle.remove()

        # 可视化第一张图片的前16个通道
        feature_maps = features[layer_name][0].cpu()
        grid = vutils.make_grid(feature_maps[:16].unsqueeze(1), nrow=4, padding=2, normalize=True)

        wandb.log({
            f"feature_maps_{layer_name}": wandb.Image(grid, caption=f"Feature Maps from {layer_name}")
        })

    def visualize_loss_landscape(self, criterion, optimizer, num_points=20):
        """可视化loss景观"""
        original_params = [p.clone() for p in self.model.parameters()]

        # 获取一个batch的数据
        images, targets = next(iter(self.train_loader))
        images, targets = images.to(self.device), targets.to(self.device)

        # 在两个随机方向上采样
        direction1 = [torch.randn_like(p) for p in original_params]
        direction2 = [torch.randn_like(p) for p in original_params]

        # 归一化方向向量
        norm1 = torch.sqrt(sum((d ** 2).sum() for d in direction1))
        norm2 = torch.sqrt(sum((d ** 2).sum() for d in direction2))
        direction1 = [d / norm1 for d in direction1]
        direction2 = [d / norm2 for d in direction2]

        alpha = np.linspace(-1, 1, num_points)
        beta = np.linspace(-1, 1, num_points)
        losses = np.zeros((num_points, num_points))

        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                # 更新参数
                for p, p0, d1, d2 in zip(self.model.parameters(), original_params, direction1, direction2):
                    p.data = p0 + a * d1 + b * d2

                outputs = self.model(images)
                loss = criterion(outputs, targets)
                losses[i, j] = loss.item()

        # 恢复原始参数
        for p, p0 in zip(self.model.parameters(), original_params):
            p.data = p0

        # 创建等高线图
        fig, ax = plt.subplots()
        cs = ax.contour(alpha, beta, losses)
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_title('Loss Landscape')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')

        wandb.log({"loss_landscape": wandb.Image(plt)})
        plt.close()

    def visualize_validation_predictions(self, images, targets, outputs):
        """可视化验证集预测结果"""
        # 获取预测标签
        _, predicted = outputs.max(1)

        # 选择一些正确和错误的预测
        correct_mask = predicted == targets
        incorrect_mask = ~correct_mask

        # 可视化正确预测
        if correct_mask.any():
            correct_images = images[correct_mask][:8]
            correct_targets = targets[correct_mask][:8]
            correct_preds = predicted[correct_mask][:8]
            self.visualize_batch(correct_images, correct_targets, correct_preds, "correct_predictions")

        # 可视化错误预测
        if incorrect_mask.any():
            incorrect_images = images[incorrect_mask][:8]
            incorrect_targets = targets[incorrect_mask][:8]
            incorrect_preds = predicted[incorrect_mask][:8]
            self.visualize_batch(incorrect_images, incorrect_targets, incorrect_preds, "incorrect_predictions")

    def visualize_grad_flow(self, named_parameters):
        """可视化梯度流"""
        ave_grads = []
        layers = []

        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())

        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(layers)), ave_grads)
        plt.xticks(np.arange(len(layers)), layers, rotation=45)
        plt.ylabel("average gradient")
        plt.title("Gradient Flow")
        plt.tight_layout()

        wandb.log({"gradient_flow": wandb.Image(plt)})
        plt.close()


def update_train_loop(train_one_epoch, visualizer):
    """更新训练循环，添加可视化"""

    def train_with_vis(model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            # 可视化训练batch
            if batch_idx % 100 == 0:
                visualizer.visualize_batch(images, targets, phase="train")
                visualizer.visualize_feature_maps(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()

            # 可视化梯度流
            if batch_idx % 100 == 0:
                visualizer.visualize_grad_flow(model.named_parameters())

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 可视化loss景观
            if batch_idx % 200 == 0:
                visualizer.visualize_loss_landscape(criterion, optimizer)

            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        return total_loss / len(train_loader), 100. * correct / total

    return train_with_vis


def update_validation_loop(validate, visualizer):
    """更新验证循环，添加可视化"""

    def validate_with_vis(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                # 定期可视化验证结果
                if batch_idx % 50 == 0:
                    visualizer.visualize_validation_predictions(images, targets, outputs)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(test_loader), 100. * correct / total

    return validate_with_vis


# Training transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

# Test transforms
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
train_dataset = CUB_200(root='CUB-200', download=True, train=True, transform=train_transform)
test_dataset = CUB_200(root='CUB-200', download=True, train=False, transform=test_transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(test_loader), 100. * correct / total


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    wandb.init(
        project="CUB-200-Classification",
        config={
            "learning_rate": 1e-3,
            "epochs": 100,
            "batch_size": 32,
            "architecture": "ResNet50v2",
            "dataset": "CUB-200"
        }
    )

    # Create model
    model = resnet50.ResNet50()
    model = model.to(device)

    # Initialize visualizer
    visualizer = TrainingVisualizer(model, train_loader, test_loader, device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Grams(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Update training and validation functions with visualization
    train_one_epoch_vis = update_train_loop(train_one_epoch, visualizer)
    validate_vis = update_validation_loop(validate, visualizer)

    # Training loop
    best_acc = 0
    for epoch in range(100):
        print(f'\nEpoch: {epoch + 1}')

        # Train with visualization
        train_loss, train_acc = train_one_epoch_vis(
            model, train_loader, criterion, optimizer, device)

        # Validate with visualization
        val_loss, val_acc = validate_vis(
            model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    wandb.finish()


if __name__ == '__main__':
    main()

