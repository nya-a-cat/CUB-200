import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from collections import defaultdict


class SemiSupervisedCUB200(Dataset):
    """
    CUB-200数据集的半监督版本，支持将指定比例的数据设为无标签
    """

    def __init__(self, root, train=True, transform=None, unlabeled_ratio=0.6):
        """
        Args:
            root: 数据集根目录
            train: 是否为训练集
            transform: 数据转换
            unlabeled_ratio: 每个类别中无标签数据的比例(0到1之间)
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.unlabeled_ratio = unlabeled_ratio

        # 加载图片和标签文件
        self.data_dir = os.path.join(root, 'CUB_200_2011')

        # 读取图片列表和对应的标签
        images_file = os.path.join(self.data_dir, 'images.txt')
        labels_file = os.path.join(self.data_dir, 'image_class_labels.txt')
        split_file = os.path.join(self.data_dir, 'train_test_split.txt')

        # 读取训练测试分割信息
        with open(split_file, 'r') as f:
            split_lines = f.readlines()
        split_dict = {line.split()[0]: int(line.split()[1]) for line in split_lines}

        # 读取标签信息
        with open(labels_file, 'r') as f:
            label_lines = f.readlines()
        label_dict = {line.split()[0]: int(line.split()[1]) - 1 for line in label_lines}  # 将标签转换为0-based

        # 读取图片路径
        with open(images_file, 'r') as f:
            image_lines = f.readlines()

        # 按类别组织数据
        class_images = defaultdict(list)
        for line in image_lines:
            img_id, img_path = line.strip().split()
            is_train = split_dict[img_id] == 1

            if is_train == self.train:  # 只处理当前split的数据
                label = label_dict[img_id]
                class_images[label].append((img_id, img_path))

        # 构建最终的数据列表
        self.images = []
        self.labels = []

        for label, img_list in class_images.items():
            # 按文件名排序
            sorted_images = sorted(img_list, key=lambda x: x[1])
            num_images = len(sorted_images)
            num_unlabeled = int(num_images * unlabeled_ratio)

            for i, (img_id, img_path) in enumerate(sorted_images):
                self.images.append(os.path.join(self.data_dir, 'images', img_path))
                # 前R%的图片设为无标签（用-1表示）
                if i < num_unlabeled:
                    self.labels.append(-1)
                else:
                    self.labels.append(label)

    def __getitem__(self, index):
        """
        返回一个数据样本
        """
        img_path = self.images[index]
        label = self.labels[index]

        # 加载并转换图片
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)