# utils.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def tensor_to_image(tensor):
    # (C,H,W) -> (H,W,C)
    img = tensor.permute(1, 2, 0)
    img = img.cpu().numpy()
    # 归一化到0~1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def visualize_feature_map(feature_map):
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')

def apply_inverse_transform_to_tensor(tensor, inverse_transform):
    # 假设 inverse_transform 可以对 Tensor 直接操作
    return inverse_transform(tensor)

def visualize_consistency(original_image, aug1_image, aug2_image,
                          fs, ft_compressed, inv_aug1, inv_aug2, num_channels=4):
    invaug1_fs = apply_inverse_transform_to_tensor(fs, inv_aug1)
    invaug2_ft_compressed = apply_inverse_transform_to_tensor(ft_compressed, inv_aug2)

    fig, axes = plt.subplots(num_channels, 7, figsize=(12, 3 * num_channels))

    for i in range(num_channels):
        # Original
        ax = axes[i, 0]
        ax.imshow(tensor_to_image(original_image))
        if i == 0:
            ax.set_title('Original', fontsize=10)
        ax.axis('off')

        # Aug1
        ax = axes[i, 1]
        ax.imshow(tensor_to_image(aug1_image))
        if i == 0:
            ax.set_title('Aug1', fontsize=10)
        ax.axis('off')

        # Aug2
        ax = axes[i, 2]
        ax.imshow(tensor_to_image(aug2_image))
        if i == 0:
            ax.set_title('Aug2', fontsize=10)
        ax.axis('off')

        # Fs
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

        # Ft_compressed
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

def consistency_loss(invaug2_Ft, invaug1_Fs):
    return F.mse_loss(invaug2_Ft, invaug1_Fs)

def get_features(model, images, layer_name):
    """
    从指定的layer_name获取输出特征
    """
    features = {}
    def hook(module, input, output):
        features[layer_name] = output.detach()

    # 找到对应模块
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)

    handle = module.register_forward_hook(hook)
    _ = model(images)
    handle.remove()
    return features[layer_name]
