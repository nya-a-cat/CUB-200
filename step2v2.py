# step2v2.py
import torch
import torch.nn as nn
import torchvision.models as models

from writing_custom_datasets import CUB_200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import (
    consistency_loss, get_features,
    visualize_consistency,  # 如果需要可视化
)

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

        # --- Visualization (take the first sample in the batch) ---
        # 这里仅做演示，如果很多张图，可以for循环
        visualize_consistency(
            original_images[0], aug1_images[0], aug2_images[0],
            Fs[0].unsqueeze(0), Ft_compressed[0].unsqueeze(0),
            inverse_aug1_transform, inverse_aug2_transform
        )

if __name__ == "__main__":
    main()
