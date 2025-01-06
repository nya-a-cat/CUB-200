import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from writing_custom_datasets import CUB_200
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class ReversibleTransform:
    """
    实现2.1要求：创建两种强度的数据增强，并记录变换参数以便后续逆变换
    aug1: 较弱的增强 (角度范围15°, 较小的平移范围)
    aug2: 较强的增强 (角度范围30°, 较大的平移范围)
    """

    def __init__(self, stronger=False):
        self.angle = None
        self.translation = None
        self.flip = None
        self.stronger = stronger

    def __call__(self, img):
        # 根据stronger参数决定增强强度
        self.angle = torch.rand(1).item() * (30 if self.stronger else 15)  # aug2使用更大的角度范围
        self.translation = (
            torch.rand(2).tolist() if self.stronger else  # aug2使用更大的平移范围
            (torch.rand(2) * 0.15).tolist()  # aug1使用较小的平移范围
        )
        self.flip = torch.rand(1).item() > 0.5

        # 应用变换并记录参数用于后续的逆变换
        if self.flip:
            img = transforms.functional.hflip(img)

        img = transforms.functional.affine(
            img,
            angle=self.angle,
            translate=self.translation,
            scale=1.0,
            shear=0,
        )

        return img

    def inverse(self, feature_map):
        """
        实现2.4要求的一部分：执行逆变换
        对增强过程中的平面变换进行逆操作
        """
        if self.translation:
            h, w = feature_map.shape[-2:]
            pixel_trans = [-t * s for t, s in zip(self.translation, (w, h))]
            feature_map = F.affine_grid(
                torch.tensor([[1, 0, -pixel_trans[0] / w],
                              [0, 1, -pixel_trans[1] / h]]).unsqueeze(0).float().to(feature_map.device),
                feature_map.shape
            )

        if self.angle:
            feature_map = torch.rot90(
                feature_map,
                k=int(round(-self.angle / 90)),
                dims=[-2, -1]
            )

        if self.flip:
            feature_map = torch.flip(feature_map, [-1])

        return feature_map


class AugmentedCUB200(Dataset):
    """
    实现2.1要求：为每个样本创建两个不同强度的增强版本
    """

    def __init__(self, original_dataset):
        self.dataset = original_dataset
        # 创建两个不同强度的变换
        self.transform1 = ReversibleTransform(stronger=False)  # aug1: 较弱的增强
        self.transform2 = ReversibleTransform(stronger=True)  # aug2: 较强的增强

        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # 应用基础变换
        img = self.base_transform(img)
        # 创建两个不同强度的增强版本
        aug1 = self.transform1(img)  # 较弱的增强版本，用于StudentNet
        aug2 = self.transform2(img)  # 较强的增强版本，用于TeacherNet

        return aug1, aug2, label, self.transform1, self.transform2


class FeatureAlignmentLoss(nn.Module):
    """
    实现2.4要求：计算特征对齐的一致性损失
    L = ||invaug2(Ft) - invaug1(Fs)||
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, student_features, teacher_features, transform1, transform2):
        # 应用逆变换以对齐特征图
        aligned_teacher = transform2.inverse(teacher_features)  # invaug2(Ft)
        aligned_student = transform1.inverse(student_features)  # invaug1(Fs)

        return self.criterion(aligned_student, aligned_teacher)


def train_student(teacher_model, student_model, train_loader, optimizer,
                  cls_criterion, consistency_criterion, device, epoch,
                  visualization_freq=100):
    """
    实现2.2-2.5要求：训练循环，包含特征提取、损失计算和模型更新
    """
    teacher_model.eval()  # 2.5要求：冻结TeacherNet
    student_model.train()

    total_loss = 0
    cls_losses = 0
    consistency_losses = 0

    for batch_idx, (aug1, aug2, labels, transform1, transform2) in enumerate(train_loader):
        aug1, aug2 = aug1.to(device), aug2.to(device)
        labels = labels.to(device)

        # 2.3要求：从TeacherNet获取特征Ft
        with torch.no_grad():  # 确保不更新教师网络
            teacher_output, teacher_features = teacher_model(aug2, return_features=True)

        # 2.2要求：从StudentNet获取特征Fs
        student_output, student_features = student_model(aug1, return_features=True)

        # 2.5要求：计算分类损失
        cls_loss = cls_criterion(student_output, labels)

        # 2.4要求：计算一致性损失
        consistency_loss = consistency_criterion(
            student_features, teacher_features, transform1, transform2
        )

        # 总损失
        loss = cls_loss + 0.5 * consistency_loss

        # 更新StudentNet
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        cls_losses += cls_loss.item()
        consistency_losses += consistency_loss.item()

        # 2.6要求：定期可视化特征对齐情况
        if batch_idx % visualization_freq == 0:
            visualize_feature_alignment(
                student_features[0], teacher_features[0],
                transform1[0], transform2[0],
                batch_idx, epoch
            )

        # 使用wandb记录训练过程
        wandb.log({
            "epoch": epoch,
            "batch": batch_idx,
            "total_loss": loss.item(),
            "classification_loss": cls_loss.item(),
            "consistency_loss": consistency_loss.item()
        }, step=epoch * len(train_loader) + batch_idx)

    return total_loss / len(train_loader)


def visualize_feature_alignment(student_feat, teacher_feat, transform1, transform2, batch_idx, epoch):
    """
    实现2.6要求：可视化特征对齐效果
    通过对比原始特征图和经过逆变换后的特征图，验证位置对应关系
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 显示变换前的特征图
    axes[0, 0].imshow(student_feat[0].cpu().detach().numpy())
    axes[0, 0].set_title('Student Features (Before)')
    axes[0, 1].imshow(teacher_feat[0].cpu().detach().numpy())
    axes[0, 1].set_title('Teacher Features (Before)')

    # 显示经过逆变换对齐后的特征图
    aligned_student = transform1.inverse(student_feat.unsqueeze(0))[0]
    aligned_teacher = transform2.inverse(teacher_feat.unsqueeze(0))[0]

    axes[1, 0].imshow(aligned_student.cpu().detach().numpy())
    axes[1, 0].set_title('Student Features (Aligned)')
    axes[1, 1].imshow(aligned_teacher.cpu().detach().numpy())
    axes[1, 1].set_title('Teacher Features (Aligned)')

    plt.tight_layout()
    wandb.log({
        "feature_alignment": wandb.Image(fig),
        "epoch": epoch,
        "batch": batch_idx
    })
    plt.close()

def main():
    # Initialize models
    teacher_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 200)
    # Load the trained teacher weights here
    teacher_model.load_state_dict(torch.load('model_checkpoints/best_model.pth')['model_state_dict'])

    student_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    student_model.fc = nn.Linear(student_model.fc.in_features, 200)

    # Add feature extraction capability
    for model in [teacher_model, student_model]:
        def forward_features(self, x, return_features=False):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            features = self.layer4(x)

            x = self.avgpool(features)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            if return_features:
                return x, features
            return x

        model.forward = forward_features.__get__(model)

    # Create datasets and dataloaders
    train_data = CUB_200(
        root='CUB-200',
        download=True,
        transform=None,  # We'll apply transforms in AugmentedCUB200
        train=True
    )

    test_data = CUB_200(
        root='CUB-200',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        train=False
    )

    # Wrap the training dataset with our augmentation wrapper
    augmented_train_data = AugmentedCUB200(train_data)

    # Create data loaders
    train_loader = DataLoader(
        augmented_train_data,
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

    # Initialize wandb
    wandb.init(
        project="knowledge-distillation",
        config={
            "architecture": "resnet18_student",
            "dataset": "CUB200",
            "epochs": 100,
        }
    )

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # 实现2.5要求：冻结教师网络参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Setup criteria and optimizer
    cls_criterion = nn.CrossEntropyLoss()
    consistency_criterion = FeatureAlignmentLoss()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_student(
            teacher_model, student_model, train_loader,
            optimizer, cls_criterion, consistency_criterion,
            device, epoch
        )

        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")


if __name__ == "__main__":
    main()