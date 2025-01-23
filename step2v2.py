# step2v2.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

from writing_custom_datasets import CUB_200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import (
    consistency_loss, get_features,
)

def visualize_inverse_transform(original_images, aug1_images, aug2_images, Fs, Ft, invaug1_Fs, invaug2_Ft, idx=0):
    """
    Visualizes the original images, augmented images, feature maps, and inverse transformed feature maps
    to verify the correctness of the inverse transformations.

    Args:
        original_images: Batch of original images.
        aug1_images: Batch of images after augmentation 1.
        aug2_images: Batch of images after augmentation 2.
        Fs: Feature maps from the student network.
        Ft: Feature maps from the teacher network.
        invaug1_Fs: Inverse transformed feature maps of Fs.
        invaug2_Ft: Inverse transformed feature maps of Ft.
        idx: Index of the sample in the batch to visualize.
    """

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    def visualize_feature_map(feature_map, title):
        # Take the mean across channels as a simple visualization
        feature_map_mean = np.mean(feature_map, axis=0)
        plt.imshow(feature_map_mean, cmap='viridis')
        plt.title(title)
        plt.colorbar()

    original_img = to_numpy(original_images[idx]).transpose(1, 2, 0)
    aug1_img = to_numpy(aug1_images[idx]).transpose(1, 2, 0)
    aug2_img = to_numpy(aug2_images[idx]).transpose(1, 2, 0)

    Fs_sample = to_numpy(Fs[idx])
    Ft_sample = to_numpy(Ft[idx])
    invaug1_Fs_sample = to_numpy(invaug1_Fs[idx])
    invaug2_Ft_sample = to_numpy(invaug2_Ft[idx])

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(original_img)
    plt.title("Original Image")

    plt.subplot(2, 4, 2)
    plt.imshow(aug1_img)
    plt.title("Aug1 Image")

    plt.subplot(2, 4, 3)
    plt.imshow(aug2_img)
    plt.title("Aug2 Image")

    plt.subplot(2, 4, 5)
    visualize_feature_map(Fs_sample, "Fs (Student Feature Map)")

    plt.subplot(2, 4, 6)
    visualize_feature_map(Ft_sample, "Ft (Teacher Feature Map)")

    plt.subplot(2, 4, 7)
    visualize_feature_map(invaug1_Fs_sample, "InvAug1(Fs)")

    plt.subplot(2, 4, 8)
    visualize_feature_map(invaug2_Ft_sample, "InvAug2(Ft)")

    plt.tight_layout()
    plt.show()

def main():
    torch.multiprocessing.freeze_support()

    # --- Hyperparameters and Configurations ---
    batch_size = 4  # Reduced batch size for visualization
    image_size = 224
    num_classes = 200
    layer_name = 'layer4'
    num_visualize_samples = 4  # Number of samples to visualize

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    cub_dataset_train = CUB_200(root='CUB-200', train=True, download=True)
    contrastive_dataloader = create_contrastive_dataloader(
        dataset=cub_dataset_train,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=batch_size,
        shuffle=False  # Disable shuffle for consistent sample selection
    )

    # --- Model Initialization ---
    student_net = models.resnet18(pretrained=True)
    teacher_net = models.resnet50(pretrained=True)
    for param in teacher_net.parameters():
        param.requires_grad = False  # Freeze TeacherNet parameters
    student_net.fc = nn.Linear(512, num_classes)
    teacher_net.fc = nn.Linear(2048, num_classes)

    # --- 1x1 Convolution for Channel Compression ---
    teacher_feature_dim = teacher_net._modules[layer_name][-1].conv3.out_channels  # Changed to conv3 for layer4 in ResNet50
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

        # --- Visualization (take the first sample in the batch) ---
        visualize_inverse_transform(original_images, aug1_images, aug2_images, Fs, Ft_compressed, invaug1_Fs, invaug2_Ft)

if __name__ == "__main__":
    main()