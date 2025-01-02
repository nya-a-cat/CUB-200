import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import resnet50v2
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

    def visualize_batch(self, images, targets, predictions=None, phase="train", caption=None):
        """可视化一个batch的图像，支持自定义标题"""
        images_cpu = self.denormalize(images.cpu())
        grid = vutils.make_grid(images_cpu[:16], nrow=4, padding=2, normalize=True)

        if caption is None:
            caption = f"{phase.capitalize()} Batch Images"
            if predictions is not None:
                caption += f" (Pred: {predictions[:16].tolist()})"

        wandb.log({
            f"{phase}_batch": wandb.Image(grid, caption=caption)
        })

    def visualize_network_flow(self, images, targets, phase="train"):
        """可视化网络流程和中间层特征图"""
        self.model.eval()
        features = {}

        # 定义要可视化的层
        layers_to_viz = ['conv1', 'bn1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']

        # 注册钩子
        handles = []
        for name, module in self.model.named_modules():
            if name in layers_to_viz:
                handles.append(module.register_forward_hook(
                    lambda m, i, o, name=name: features.update({name: o})
                ))

        # 前向传播
        with torch.no_grad():
            outputs = self.model(images.to(self.device))

        # 移除钩子
        for handle in handles:
            handle.remove()

        # 创建可视化图
        batch_size = min(8, images.size(0))
        fig = plt.figure(figsize=(20, 4 * batch_size))

        for sample_idx in range(batch_size):
            # 绘制输入图像
            ax = plt.subplot(batch_size, len(layers_to_viz) + 1, sample_idx * (len(layers_to_viz) + 1) + 1)
            img = self.denormalize(images[sample_idx]).cpu()
            plt.imshow(img.permute(1, 2, 0))
            if sample_idx == 0:
                ax.set_title('Input')
            ax.axis('off')

            # 绘制中间层特征图
            for layer_idx, layer_name in enumerate(layers_to_viz[:-1], 2):
                ax = plt.subplot(batch_size, len(layers_to_viz) + 1,
                                 sample_idx * (len(layers_to_viz) + 1) + layer_idx)
                feature = features[layer_name][sample_idx].cpu()

                if layer_name == 'avgpool':
                    # 修改这里的处理方式，确保特征图是2D的
                    feature = feature.view(1, -1)  # 重塑为2D
                    plt.imshow(feature, cmap='viridis')
                else:
                    # 对于其他层，保持原来的处理方式
                    plt.imshow(feature.mean(0), cmap='viridis')

                if sample_idx == 0:
                    ax.set_title(layer_name)
                ax.axis('off')

            # 绘制输出预测
            ax = plt.subplot(batch_size, len(layers_to_viz) + 1,
                             (sample_idx + 1) * (len(layers_to_viz) + 1))
            probs = F.softmax(outputs[sample_idx], dim=0)
            top_k = torch.topk(probs, k=5)
            plt.bar(range(5), top_k.values.cpu())
            plt.xticks(range(5), [f'Class {i}' for i in top_k.indices.cpu()], rotation=45)
            if sample_idx == 0:
                ax.set_title('Predictions')

        plt.tight_layout()
        wandb.log({f"{phase}_network_flow": wandb.Image(plt)})
        plt.close()

    def visualize_validation_predictions(self, images, targets, outputs):
        """可视化验证集预测结果，包括正确和错误的预测"""
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        correct_mask = predicted == targets
        incorrect_mask = ~correct_mask

        # 可视化正确预测
        if correct_mask.any():
            correct_indices = torch.where(correct_mask)[0][:8]
            correct_images = images[correct_indices]
            correct_conf = confidence[correct_indices]

            caption = (f"Correct Predictions\n"
                       f"Confidence: {[f'{conf:.2f}' for conf in correct_conf.cpu().numpy()]}")
            self.visualize_batch(correct_images, None, phase="correct_predictions", caption=caption)

        # 可视化错误预测
        if incorrect_mask.any():
            incorrect_indices = torch.where(incorrect_mask)[0][:8]
            incorrect_images = images[incorrect_indices]
            incorrect_targets = targets[incorrect_indices]
            incorrect_preds = predicted[incorrect_indices]
            incorrect_conf = confidence[incorrect_indices]

            caption = (f"Incorrect Predictions\n"
                       f"True: {incorrect_targets.cpu().numpy()}\n"
                       f"Pred: {incorrect_preds.cpu().numpy()}\n"
                       f"Conf: {[f'{conf:.2f}' for conf in incorrect_conf.cpu().numpy()]}")
            self.visualize_batch(incorrect_images, None, phase="incorrect_predictions", caption=caption)


def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化wandb
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

    # 数据转换
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomRotation(degrees=10),
    #     transforms.RandAugment(num_ops=2, magnitude=7),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     transforms.RandomErasing(p=0.2)
    # ])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 首先调整大小到稍大尺寸
        transforms.RandomCrop(224),  # 随机裁剪到目标尺寸
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 降低颜色抖动的强度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据加载
    train_dataset = CUB_200(root='CUB-200', download=True, train=True, transform=train_transform)
    test_dataset = CUB_200(root='CUB-200', download=True, train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 模型初始化
    model = resnet50v2.ResNet50()
    model = model.to(device)

    # 初始化可视化器
    visualizer = TrainingVisualizer(model, train_loader, test_loader, device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Grams(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 训练循环
    best_acc = 0
    for epoch in range(100):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            if batch_idx % 100 == 0:
                visualizer.visualize_batch(images, targets, phase="train")
                visualizer.visualize_network_flow(images, targets, phase="train")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                # 获取预测结果
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)

                # 构建caption，包含真实标签和预测标签
                caption = f"Test Batch {batch_idx}\n"
                caption += f"True labels: {targets[:16].cpu().tolist()}\n"
                caption += f"Predicted: {predicted[:16].cpu().tolist()}"

                # 可视化当前batch
                visualizer.visualize_batch(images, targets, predicted, phase=f"test_batch_{batch_idx}", caption=caption)

                if batch_idx % 50 == 0:
                    visualizer.visualize_validation_predictions(images, targets, outputs)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # 更新学习率
        scheduler.step()

        # 计算指标
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total

        # 记录到wandb
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    wandb.finish()


if __name__ == '__main__':
    main()

