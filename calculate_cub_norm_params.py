import torch
import torchvision.transforms as transforms
from cub_dataset import CUB_200  # 确保 cub_dataset.py 文件在同一目录下

from torch.utils.data import DataLoader

def calculate_mean_std(dataset):
    """计算数据集的均值和标准差。

    Args:
        dataset: PyTorch Dataset 对象。

    Returns:
        mean: 各个通道的均值 (tuple of floats)。
        std: 各个通道的标准差 (tuple of floats)。
    """
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for batch in dataloader:
        data = batch[0] if isinstance(batch, (list, tuple)) else batch # 假设图像是 batch 的第一个元素
        if isinstance(data, list): # 处理 contrastive_batch 的情况，虽然这里应该不会遇到
            data = data[0] # 取 'original' 图像
        data = data.float() / 255.0 # 假设你的图像像素值在 0-255 范围内，先缩放到 0-1

        for d in range(3):
            mean[d] += torch.sum(torch.mean(data[:, d, :, :], dim=[1, 2]))
            std[d] += torch.sum(torch.std(data[:, d, :, :], dim=[1, 2]))
        total_samples += data.size(0) * len(dataloader)

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist()


if __name__ == '__main__':
    # --- 配置 ---
    config = {
        "image_size": 224,
        "unlabeled_ratio": 0.6,
        "batch_size": 64
    }

    # --- 创建 Dataset (使用你自定义的 CUB_200 Dataset 类) ---
    train_dataset_stats = CUB_200(
        root='CUB-200', # 替换为你的 CUB 数据集根目录
        train=True,
        transform=transforms.ToTensor(), #  ToTensor() 是必须的，为了计算均值和标准差
        download=True # 如果数据集不存在，可以下载
    )

    # --- 计算均值和标准差 ---
    mean, std = calculate_mean_std(train_dataset_stats)
    print(f"CUB Dataset Mean: {mean}, Std: {std}")

    print("归一化参数计算完成！")
    print("请将上述 Mean 和 Std 值复制到你的训练代码中 Normalize Transform 的参数里。")