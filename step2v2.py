import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# --- Data Loading Modules ---
from writing_custom_datasets import CUB_200

class ContrastiveDataset(Dataset):
    """
    Wraps a base dataset to produce pairs of differently augmented views of each image.
    Includes the original (unaugmented) image for potential use.
    """
    def __init__(self, base_dataset, augmentation1, augmentation2):
        self.base_dataset = base_dataset
        self.aug1 = augmentation1
        self.aug2 = augmentation2
        self.to_tensor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {
            'aug1': self.aug1(image),
            'aug2': self.aug2(image),
            'label': label,
            'original': self.to_tensor(image)
        }

    def __len__(self):
        return len(self.base_dataset)

def create_contrastive_dataloader(dataset, aug1, aug2, batch_size=32, shuffle=True, num_workers=0):
    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)
    return DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# --- Transformation Modules ---
def get_augmentation_transforms(size=224):
    common_transforms = [
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    aug1 = transforms.Compose([*common_transforms, transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)])
    aug2 = transforms.Compose([*common_transforms, transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2), transforms.RandomRotation(30)])
    return aug1, aug2

def get_inverse_transforms():
    inv_aug1 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
    inv_aug2 = transforms.Compose([transforms.RandomRotation((-30, -30)), transforms.RandomHorizontalFlip(p=1.0)])
    return inv_aug1, inv_aug2

# --- Visualization Modules ---
def tensor_to_image(tensor):
    img = tensor.permute(1, 2, 0)
    img = img.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def visualize_feature_map(feature_map):
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')

def apply_inverse_transform_to_tensor(tensor, inverse_transform):
    return inverse_transform(tensor)

def visualize_consistency(original_image, aug1_image, aug2_image, fs, ft_compressed, inv_aug1, inv_aug2, num_channels=4):
    invaug1_fs = apply_inverse_transform_to_tensor(fs, inv_aug1)
    invaug2_ft_compressed = apply_inverse_transform_to_tensor(ft_compressed, inv_aug2)

    fig, axes = plt.subplots(num_channels, 7, figsize=(12, 3 * num_channels)) # 调整 figsize 以使图表更紧凑

    for i in range(num_channels):
        # Original Image (only in the first row)
        ax = axes[i, 0]
        ax.imshow(tensor_to_image(original_image))
        if i == 0:
            ax.set_title('Original', fontsize=10) # 减小字体大小
        ax.axis('off')

        # Aug1 Image (only in the first row)
        ax = axes[i, 1]
        ax.imshow(tensor_to_image(aug1_image))
        if i == 0:
            ax.set_title('Aug1', fontsize=10)
        ax.axis('off')

        # Aug2 Image (only in the first row)
        ax = axes[i, 2]
        ax.imshow(tensor_to_image(aug2_image))
        if i == 0:
            ax.set_title('Aug2', fontsize=10)
        ax.axis('off')

        # Fs from StudentNet
        ax = axes[i, 3]
        ax.imshow(fs[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('Fs', fontsize=10)
        ax.axis('off')

        # invaug1(Fs)
        ax = axes[i, 4]
        ax.imshow(invaug1_fs[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('invaug1(Fs)', fontsize=10)
        ax.axis('off')

        # Ft_compressed from TeacherNet
        ax = axes[i, 5]
        ax.imshow(ft_compressed[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('Ft_compressed', fontsize=10)
        ax.axis('off')

        # invaug2(Ft_compressed)
        ax = axes[i, 6]
        ax.imshow(invaug2_ft_compressed[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('invaug2(Ft_compressed)', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- Loss Modules ---
def consistency_loss(invaug2_Ft, invaug1_Fs):
    return F.mse_loss(invaug2_Ft, invaug1_Fs)

# --- Model Feature Extraction ---
def get_features(model, images, layer_name):
    features = {}
    def hook(module, input, output):
        features[layer_name] = output.detach()
    layer = None
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    layer = module

    handle = layer.register_forward_hook(hook)
    _ = model(images)
    handle.remove()
    return features[layer_name]

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # --- Hyperparameters and Configurations ---
    batch_size = 2  # Reduced batch size for visualization
    image_size = 224
    num_classes = 200
    layer_name = 'layer4'
    num_visualize_samples = 2  # Number of samples to visualize

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    cub_dataset_train = CUB_200(root='CUB-200', train=True, download=True)
    contrastive_dataloader = create_contrastive_dataloader(
        dataset=cub_dataset_train,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=batch_size,
        shuffle=False # Disable shuffle for consistent sample selection
    )

    # --- Model Initialization ---
    student_net = models.resnet18(pretrained=True)
    teacher_net = models.resnet18(pretrained=True)
    for param in teacher_net.parameters():
        param.requires_grad = False  # Freeze TeacherNet parameters

    # --- 1x1 Convolution for Channel Compression ---
    teacher_feature_dim = teacher_net._modules[layer_name][-1].conv2.out_channels
    student_feature_dim = student_net._modules[layer_name][-1].conv2.out_channels
    compression_layer = nn.Conv2d(teacher_feature_dim, student_feature_dim, kernel_size=1)

    # --- Visualize Consistency for Selected Samples ---
    data_iter = iter(contrastive_dataloader)
    for _ in range(num_visualize_samples):
        contrastive_batch = next(data_iter)
        original_images = contrastive_batch['original']
        aug1_images = contrastive_batch['aug1']
        aug2_images = contrastive_batch['aug2']

        # --- Feature Extraction ---
        Fs = get_features(student_net, aug1_images, layer_name)
        Ft = get_features(teacher_net, aug2_images, layer_name)

        # --- Channel Compression ---
        Ft_compressed = compression_layer(Ft)

        # --- Apply Inverse Augmentation to Features ---
        invaug1_Fs = inverse_aug1_transform(Fs)
        invaug2_Ft = inverse_aug2_transform(Ft_compressed)

        # --- Consistency Loss ---
        loss = consistency_loss(invaug2_Ft, invaug1_Fs)
        print(f"Consistency Loss: {loss.item()}")

        # Visualize for the first sample in the batch
        visualize_consistency(original_images[0], aug1_images[0], aug2_images[0], Fs[0].unsqueeze(0), Ft_compressed[0].unsqueeze(0), inverse_aug1_transform, inverse_aug2_transform)

        print("请分析可视化结果，观察增强、特征图扭曲和逆变换的效果，以及invaug1(Fs)和invaug2(Ft_compressed)的空间一致性。")