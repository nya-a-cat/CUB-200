import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch
import time
import os

def check_dataset(train_data, test_data):
    """
    检查数据集是否存在问题：
    1. 标签范围是否从 0 到 199，对齐模型输出的类索引。
    2. 类别分布是否平衡。
    3. 训练集和测试集是否有重叠。
    """
    # 检查标签范围
    train_labels = [label for _, label in train_data]
    test_labels = [label for _, label in test_data]

    # 检查标签的最小值和最大值
    train_min, train_max = min(train_labels), max(train_labels)
    test_min, test_max = min(test_labels), max(test_labels)
    print(f"训练集标签范围：{train_min} - {train_max}")
    print(f"测试集标签范围：{test_min} - {test_max}")

    # 检查标签范围是否从 0 开始
    if train_min != 0 or test_min != 0 or train_max != 199 or test_max != 199:
        print("标签范围异常！请确保标签范围从 0 到 199，并对标签进行重新编码。")

    # 检查类别分布
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)

    print(f"训练集类别分布：{train_label_counts}")
    print(f"测试集类别分布：{test_label_counts}")

    # 可视化类别分布
    plt.figure(figsize=(12, 6))
    plt.bar(train_label_counts.keys(), train_label_counts.values(), alpha=0.6, label='Train')
    plt.bar(test_label_counts.keys(), test_label_counts.values(), alpha=0.6, label='Test')
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.legend()
    plt.show()

    # 检查训练集和测试集是否有重叠
    train_images = {train_data.samples[i][0] for i in range(len(train_data))}
    test_images = {test_data.samples[i][0] for i in range(len(test_data))}

    overlapping_images = train_images & test_images
    if overlapping_images:
        print(f"训练集和测试集有 {len(overlapping_images)} 个重叠样本！")
        print("重叠样本示例：", list(overlapping_images)[:5])
    else:
        print("训练集和测试集没有重叠样本。")


# Data processing code remains the same
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


train_data = CUB_200(root='CUB-200', download=True, transform=train_transform, train=True)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform, train=False)

check_dataset(train_data, test_data)
