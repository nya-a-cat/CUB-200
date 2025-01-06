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


def get_aug1_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
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


def get_aug2_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=10),
            transforms.RandomRotation(30),
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


class CUB_200_DualAugDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform1=None, transform2=None, train=True):
        self.data = CUB_200(root=root, train=train)  # 假设 CUB_200 类已经存在
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本图片和标签
        img, label = self.data[idx]

        # 应用两个不同的增强方式
        img_aug1 = self.transform1(img) if self.transform1 else img
        img_aug2 = self.transform2(img) if self.transform2 else img

        return img_aug1, img_aug2, label


def main():
    # 基础设置
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_data = CUB_200_DualAugDataset(
        root='CUB-200',
        transform1=get_aug1_transforms(train=True),  # Define the augmentation transformations
        transform2=get_aug2_transforms(train=True),
        train=True
    )
    test_data = CUB_200_DualAugDataset(
        root='CUB-200',
        transform1=get_aug1_transforms(train=False),
        transform2=get_aug2_transforms(train=False),
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
    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load('model_checkpoints/best_model.pth')['model_state_dict'])
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    model = model.to(device)

    # 可选：初始化wandb
    wandb.init(project="cub-200-classification", name="resnet50-optimized")

    # Training and validation code (with feature map extraction)
    model.eval()  # Set the model to evaluation mode
    for images, labels in train_loader:
        images = images.to(device)

        # Apply the first augmentation (aug1) transformation
        augmented_images = get_aug1_transforms(train=False)(images)  # Apply aug1 to the images

        # Forward pass through the model
        x = augmented_images
        x = model.conv1(x)  # Pass through the initial conv layer
        x = model.bn1(x)  # Pass through batch norm
        x = model.relu(x)  # Pass through ReLU
        x = model.maxpool(x)  # Max pooling layer

        x = model.layer1(x)  # Pass through ResNet layer1
        x = model.layer2(x)  # Pass through ResNet layer2
        x = model.layer3(x)  # Pass through ResNet layer3
        Fs = model.layer4(x)  # Final feature map from ResNet layer4

        # Optionally, detach the feature map from the graph and move to CPU
        Fs = Fs.detach().cpu()  # Detach to avoid gradients, move to CPU

        print(Fs.shape)  # Print the shape of the feature map

        # Further processing or logging with wandb could be done here
        # wandb.log({"feature_map": Fs})

        # Optionally, stop after the first batch for testing purposes
        break

    # 训练和验证代码继续...

if __name__ == '__main__':
    main()