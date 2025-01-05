import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch
import time
import os

# Data processing code remains the same
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomRotation(10),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


transform_aug1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomRotation(10),  # 保留小角度旋转
])

transform_aug2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    # 1. 几何变换类(可逆)
    transforms.RandomRotation(45),  # 旋转范围[-45,45]度
    transforms.RandomAffine(
        degrees=45,  # 仿射旋转45度
        translate=(0.2, 0.2),  # 平移范围20%
        scale=(0.8, 1.2),  # 缩放范围更合理
        shear=(-15, 15, -15, 15)  # 适度的剪切变换
    ),

    # 2. 颜色变换类(可逆)
    transforms.ColorJitter(
        brightness=0.3,  # 亮度变化
        contrast=0.3,  # 对比度变化
        saturation=0.3,  # 饱和度变化
        hue=0.1  # 色相变化(保持较小以确保可逆)
    ),

    # 3. 翻转(可逆)
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 垂直翻转

])



train_arug1_data = CUB_200(root='CUB-200', download=True, transform=transform_aug1)
train_arug2_data = CUB_200(root='CUB-200', download=True, transform=transform_aug2)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform)

train_arug1_loader = torch.utils.data.DataLoader(
    train_arug1_data,
    batch_size=200,
    shuffle=None,
    num_workers=8,
    prefetch_factor=2)
train_arug2_loader = torch.utils.data.DataLoader(
    train_arug2_data,
    batch_size=200,
    shuffle=None,
    num_workers=8,
    prefetch_factor=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=200, shuffle=False, num_workers=8, prefetch_factor=2)


# 定义模型
teacher = models.resnet50(weights=None)
teacher.fc = torch.nn.Linear(teacher.fc.in_features, 200)

student = models.resnet18(weights=None)
student.fc = torch.nn.Linear(student.fc.in_features, 200)

# 2.1
student(train_arug1_loader)
Fs = student.fc
teacher(train_arug2_loader)
Ft = teacher.fc
Ft = torch.nn.Conv2d(in_channels=Ft.shape[1],  # FT的通道数
                         out_channels=Fs.out_features,   # FS的通道数
                         kernel_size=1)


def consistency_loss(Ft, Fs, invaug1, invaug2):
    # 应用逆增强
    Ft_restore = invaug2(Ft)  # teacher特征逆增强
    Fs_restore = invaug1(Fs)  # student特征逆增强

    # 计算L2距离
    loss = torch.nn.functional.mse_loss(Ft_restore, Fs_restore)
    # 或者用L1距离
    # loss = torch.nn.functional.l1_loss(Ft_restore, Fs_restore)

    return loss


# 加载模型权重
checkpoint = torch.load('model_checkpoints/best_model.pth')
teacher.load_state_dict(checkpoint['model_state_dict'])

# 移到GPU(如果需要)
if torch.cuda.is_available():
    model.cuda()

# 设置为评估模式
teacher.eval()

# 使用模型进行预测
with torch.no_grad():
    # 这里放入你的推理代码
    outputs = teacher(input_images)

