import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch
import time
import os
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb  # 可选：用于实验追踪


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 数据增强和预处理
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


# 验证函数
def validate(model, test_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_total = 0
    valid_correct = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            valid_total += batch_labels.size(0)
            valid_correct += (predicted == batch_labels).sum().item()

    avg_valid_loss = valid_loss / len(test_loader)
    valid_accuracy = 100 * valid_correct / valid_total

    return avg_valid_loss, valid_accuracy


def main():
    # 基础设置
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_data = CUB_200(
        root='CUB-200',
        download=True,
        transform=get_transforms(train=True),
        train=True
    )
    test_data = CUB_200(
        root='CUB-200',
        download=True,
        transform=get_transforms(train=False),
        train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # 模型设置
    model = models.resnet50(weights='IMAGENET1K_V2')
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    model = model.to(device)

    # 损失函数和优化器设置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], lr=1e-5, weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    scaler = GradScaler()

    # 训练设置
    num_epochs = 100
    patience = 15
    min_improvement = 0.001
    best_valid_loss = float('inf')
    patience_counter = 0
    best_accuracy = 0.0

    # 创建保存模型的目录
    os.makedirs('model_checkpoints', exist_ok=True)

    # 可选：初始化wandb
    wandb.init(project="cub-200-classification", name="resnet50-optimized")

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # 验证阶段
        valid_loss, valid_accuracy = validate(model, test_loader, criterion, device)

        # 学习率调整
        scheduler.step()

        # 打印统计信息
        print(f'Epoch: {epoch + 1:3d}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        # 可选：记录到wandb
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # 早停检查
        if valid_loss < best_valid_loss * (1 - min_improvement):
            best_valid_loss = valid_loss
            patience_counter = 0

            # 保存最佳模型
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'valid_loss': valid_loss,
                    'accuracy': valid_accuracy,
                }, f'model_checkpoints/best_model.pth')
                print(f'Saved new best model with accuracy: {valid_accuracy:.2f}%')
        else:
            patience_counter += 1

        # 检查是否应该停止训练
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # 加载最佳模型进行最终评估
    checkpoint = torch.load('model_checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_accuracy = validate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    # 可选：结束wandb追踪
    wandb.finish()


if __name__ == '__main__':
    main()