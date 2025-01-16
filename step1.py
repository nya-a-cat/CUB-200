import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200  # 导入自定义的 CUB-200 数据集类
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
            transforms.Resize((256, 256)),  # 将图像大小调整到 256x256
            transforms.RandomCrop(224),  # 随机裁剪出 224x224 的区域
            transforms.RandomHorizontalFlip(),  # 以 0.5 的概率水平翻转图像
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 随机调整亮度、对比度和饱和度
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 随机进行仿射变换，包括旋转和位移
            transforms.ToTensor(),  # 将 PIL 图像或 NumPy ndarray 转换为 Tensor，并将像素值缩放到 [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 使用 ImageNet 的均值和标准差进行标准化
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),  # 将图像大小调整到 256x256
            transforms.CenterCrop(224),  # 从中心裁剪出 224x224 的区域
            transforms.ToTensor(),  # 将 PIL 图像或 NumPy ndarray 转换为 Tensor，并将像素值缩放到 [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 使用 ImageNet 的均值和标准差进行标准化
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
    # 下载 CUB-200 数据集，并使用自定义的 CUB_200 类加载训练集
    train_data = CUB_200(
        root='CUB-200',  # 数据集存放的根目录
        download=True,  # 如果数据集不存在则下载
        transform=get_transforms(train=True),  # 应用训练集的数据增强和预处理
        train=True  # 指定加载训练集
    )
    # 下载 CUB-200 数据集，并使用自定义的 CUB_200 类加载测试集
    test_data = CUB_200(
        root='CUB-200',  # 数据集存放的根目录
        download=True,  # 如果数据集不存在则下载
        transform=get_transforms(train=False),  # 应用测试集的预处理
        train=False  # 指定加载测试集
    )

    # 将加载的训练数据集放入 PyTorch 的 DataLoader 中，用于批量加载和打乱数据
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,  # 批量大小
        shuffle=True,  # 每个 epoch 开始时打乱数据
        num_workers=8,  # 使用 8 个子进程加载数据
        pin_memory=True,  # 将数据加载到 CUDA 的固定内存中，加速数据传输到 GPU
        prefetch_factor=2  # 每个 worker 预取的 batch 数量
    )
    # 将加载的测试数据集放入 PyTorch 的 DataLoader 中
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,  # 批量大小
        shuffle=False,  # 测试集不需要打乱
        num_workers=8,  # 使用 8 个子进程加载数据
        pin_memory=True,  # 将数据加载到 CUDA 的固定内存中，加速数据传输到 GPU
        prefetch_factor=2  # 每个 worker 预取的 batch 数量
    )

    # 模型设置
    # 加载预训练的 ResNet50 模型，使用在 ImageNet 上预训练的权重
    model = models.resnet50(weights='IMAGENET1K_V2')
    # 修改 ResNet50 的最后一层全连接层，使其输出 200 个类别（CUB-200 数据集的类别数）
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    # 将模型移动到 GPU (如果可用)
    model = model.to(device)

    # 损失函数和优化器设置
    criterion = torch.nn.CrossEntropyLoss()  # 使用交叉熵损失函数，适用于多分类任务
    # 使用 AdamW 优化器，并为不同的层设置不同的学习率
    optimizer = torch.optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': 1e-4},  # ResNet50 的 layer4 层使用较小的学习率
        {'params': model.fc.parameters(), 'lr': 1e-3}      # 全连接层使用较大的学习率
    ], lr=1e-5, weight_decay=0.01)  # 基础学习率设置为 1e-5，权重衰减为 0.01

    # 使用余弦退火学习率调度器，随着训练进行逐步降低学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    # 使用 GradScaler 进行混合精度训练，以加速训练并减少 GPU 内存占用
    scaler = GradScaler()

    # 训练设置
    num_epochs = 100
    patience = 15  # 早停的耐心值，当验证集损失连续 patience 个 epoch 没有下降时停止训练
    min_improvement = 0.001  # 验证集损失需要至少下降 min_improvement 才会重置耐心计数器
    best_valid_loss = float('inf')
    patience_counter = 0
    best_accuracy = 0.0

    # 创建保存模型的目录
    os.makedirs('model_checkpoints', exist_ok=True)

    # 可选：初始化wandb
    wandb.init(project="cub-200-classification", name="resnet50-optimized")

    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch_images, batch_labels in pbar:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()  # 清空之前的梯度

            with autocast():  # 启用自动混合精度
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

            scaler.scale(loss).backward()  # 反向传播，使用 scaler 缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)  # 更新优化器的参数
            scaler.update()  # 更新 scaler 的状态

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
    print(f"Final Test Accuracy: {final_accuracy:.2f}%") # 在测试集上取得比较好的效果

    # 可选：结束wandb追踪
    wandb.finish()

if __name__ == '__main__':
    main()