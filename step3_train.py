import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as transforms

from semi_supervised_dataset import SemiSupervisedCUB200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import consistency_loss, get_features


def main():
    torch.multiprocessing.freeze_support()

    # --- Hyperparameters and Configurations ---
    batch_size = 200
    image_size = 224
    num_classes = 200
    layer_name = 'layer4'
    num_visualize_samples = 4
    unlabeled_ratio = 0.6  # 60%的无标签数据

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    # 使用新的半监督数据集
    train_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=True,
        transform=transforms.ToDtype,
        unlabeled_ratio=unlabeled_ratio
    )

    contrastive_train_dataloader = create_contrastive_dataloader(
        dataset=train_dataset,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # 测试集保持不变，使用原始标签
    test_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=False,
        transform=transforms.ToDtype,
        unlabeled_ratio=0.0  # 测试集保持全标签
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # --- 模型初始化和其他逻辑保持不变 ---
    student_net = models.resnet18(pretrained=True)
    teacher_net = models.resnet18(pretrained=True)
    for param in teacher_net.parameters():
        param.requires_grad = False
    student_net.fc = nn.Linear(512, num_classes)
    teacher_net.fc = nn.Linear(512, num_classes)

    # --- 1x1 卷积用于通道压缩 ---
    teacher_feature_dim = teacher_net._modules[layer_name][-1].conv2.out_channels
    student_feature_dim = student_net._modules[layer_name][-1].conv2.out_channels
    compression_layer = nn.Conv2d(teacher_feature_dim, student_feature_dim, kernel_size=1)

    # --- 处理数据批次时需要考虑无标签数据 ---
    data_iter = iter(contrastive_train_dataloader)
    for _ in range(num_visualize_samples):
        contrastive_batch = next(data_iter)
        original_images = contrastive_batch['original']
        aug1_images = contrastive_batch['aug1']
        aug2_images = contrastive_batch['aug2']
        labels = contrastive_batch['label']  # 包含-1表示的无标签数据

        # 特征提取和一致性损失计算保持不变
        Fs = get_features(student_net, aug1_images, layer_name)
        Ft = get_features(teacher_net, aug2_images, layer_name)
        Ft_compressed = compression_layer(Ft)
        invaug1_Fs = inverse_aug1_transform(Fs)
        invaug2_Ft = inverse_aug2_transform(Ft_compressed)
        loss = consistency_loss(invaug2_Ft, invaug1_Fs)
        print(f"Consistency Loss: {loss.item()}")


if __name__ == "__main__":
    main()