import os
from pathlib import Path
from PIL import Image

import pandas as pd
import torch
import torchvision.datasets.utils as dataset_utils
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict, Iterator, Optional, Tuple


class CUB_200(VisionDataset):
    """
    CUB-200-2011 Dataset

    The Caltech-UCSD Birds-200-2011 dataset includes:
    - 11,788 images of 200 bird species
    - Annotations: bounding boxes, part locations, and attribute labels
    """

    base_folder = 'CUB_200_2011'
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = 'xxx'  # 需要填入正确的MD5值

    def __init__(self, root: str, train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 load_bbox: bool = False,
                 load_parts: bool = False,
                 load_attributes: bool = False,
                 download: bool = False):
        """
        Args:
            root: 数据集根目录
            train: 是否加载训练集
            transform: 图像变换
            target_transform: 标签变换
            load_bbox: 是否加载边界框标注
            load_parts: 是否加载部位标注
            load_attributes: 是否加载属性标注
            download: 是否下载数据集
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = Path(root)
        self.train = train
        self.load_bbox = load_bbox
        self.load_parts = load_parts
        self.load_attributes = load_attributes

        if download:
            self._download_and_extract()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # 加载基本数据
        self._load_basic_info()

        # 初始化self.data和self.targets
        self.data = self.filtered_info
        self.targets = [label - 1 for label in self.filtered_labels]  # 转换为0-based索引

        # 可选：加载额外信息
        if load_bbox:
            self.bbox_info = self._load_bounding_boxes()
        if load_parts:
            self.parts_info = self._load_parts()
        if load_attributes:
            self.attributes_info = self._load_attributes()

    def _load_basic_info(self):
        """加载基本的图像和标签信息"""
        # 加载图像路径
        image_paths_df = pd.read_csv(
            self.root / self.base_folder / 'images.txt',
            sep=' ',
            names=['image_id', 'filepath'],
            index_col='image_id'
        )

        # 加载训练/测试集划分信息
        train_test_df = pd.read_csv(
            self.root / self.base_folder / 'train_test_split.txt',
            sep=' ',
            names=['image_id', 'is_training_image'],
            index_col='image_id'
        )

        # 加载类别标签
        labels_df = pd.read_csv(
            self.root / self.base_folder / 'image_class_labels.txt',
            sep=' ',
            names=['image_id', 'class_id'],
            index_col='image_id'
        )

        # 加载类别名称并转换为列表
        classes_df = pd.read_csv(
            self.root / self.base_folder / 'classes.txt',
            sep=' ',
            names=['class_id', 'class_name']
        )
        self.classes = classes_df['class_name'].tolist()  # 将类别名称转换为列表

        # 合并所有信息
        self.data_info = image_paths_df.join([train_test_df, labels_df])

        # 根据训练/测试标志筛选数据
        mask = self.data_info['is_training_image'] == (1 if self.train else 0)
        self.filtered_info = self.data_info[mask]

        # 提取路径和标签
        self.filtered_paths = self.filtered_info['filepath'].values
        self.filtered_labels = self.filtered_info['class_id'].values

    def _load_bounding_boxes(self) -> Dict:
        """加载边界框信息"""
        bbox_df = pd.read_csv(
            self.root / self.base_folder / 'bounding_boxes.txt',
            sep=' ',
            names=['image_id', 'x', 'y', 'width', 'height']
        )

        return {
            row['image_id']: {
                'x': row['x'],
                'y': row['y'],
                'width': row['width'],
                'height': row['height']
            }
            for _, row in bbox_df.iterrows()
            if row['image_id'] in self.filtered_info.index
        }

    def _load_parts(self) -> Dict:
        """加载鸟类部位信息"""
        # 读取部位名称
        parts_df = pd.read_csv(
            self.root / self.base_folder / 'parts/parts.txt',
            sep=' ',
            names=['part_id', 'part_name']
        )

        # 读取部位坐标
        parts_locs_df = pd.read_csv(
            self.root / self.base_folder / 'parts/part_locs.txt',
            sep=' ',
            names=['image_id', 'part_id', 'x', 'y', 'visible']
        )

        # 将部位信息组织成字典
        parts_dict = {}
        for image_id in self.filtered_info.index:
            image_parts = parts_locs_df[parts_locs_df['image_id'] == image_id]
            parts_dict[image_id] = {
                row['part_id']: {
                    'name': parts_df.loc[row['part_id'], 'part_name'],
                    'x': row['x'],
                    'y': row['y'],
                    'visible': row['visible']
                }
                for _, row in image_parts.iterrows()
            }
        return parts_dict

    def _load_attributes(self) -> Dict:
        """加载属性信息"""
        # 读取属性名称
        attributes_df = pd.read_csv(
            self.root / self.base_folder / 'attributes/attributes.txt',
            sep=' ',
            names=['attr_id', 'attribute_name']
        )

        # 读取属性标签
        attr_labels_df = pd.read_csv(
            self.root / self.base_folder / 'attributes/image_attribute_labels.txt',
            sep=' ',
            names=['image_id', 'attr_id', 'is_present', 'certainty', 'time']
        )

        # 将属性信息组织成字典
        attr_dict = {}
        for image_id in self.filtered_info.index:
            image_attrs = attr_labels_df[attr_labels_df['image_id'] == image_id]
            attr_dict[image_id] = {
                row['attr_id']: {
                    'name': attributes_df.loc[row['attr_id'], 'attribute_name'],
                    'is_present': row['is_present'],
                    'certainty': row['certainty'],
                    'time': row['time']
                }
                for _, row in image_attrs.iterrows()
            }
        return attr_dict

    def _check_exists(self) -> bool:
        """检查数据集文件是否存在"""
        return (self.root / self.base_folder).exists()

    def _download_and_extract(self) -> None:
        """下载并解压数据集"""
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        dataset_utils.download_and_extract_archive(
            url=self.url,
            download_root=str(self.root),
            filename=self.filename,
            md5=self.tgz_md5,
            remove_finished=True
        )

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.filtered_paths)

    # def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
    #     """获取指定索引的样本"""
    #     # 获取图像ID
    #     image_id = self.filtered_info.index[index]
    #
    #     # 加载图像
    #     img_path = self.root / self.base_folder / 'images' / self.filtered_paths[index]
    #     image = Image.open(img_path).convert('RGB')
    #
    #     # 获取标签（转换为0-based索引）
    #     label = self.filtered_labels[index] - 1
    #
    #     # 应用变换
    #     if self.transform:
    #         image = self.transform(image)
    #     if self.target_transform:
    #         label = self.target_transform(label)
    #
    #     sample = (image, label)
    #
    #     # 添加额外信息
    #     extra_info = {}
    #
    #     if self.load_bbox:
    #         extra_info['bbox'] = self.bbox_info[image_id]
    #
    #     if self.load_parts:
    #         extra_info['parts'] = self.parts_info[image_id]
    #
    #     if self.load_attributes:
    #         extra_info['attributes'] = self.attributes_info[image_id]
    #
    #     if extra_info:
    #         sample = sample + (extra_info,)
    #
    #     return sample

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
        """
        image_id = self.filtered_info.index[index]
        img_path = self.root / self.base_folder / "images" / self.filtered_paths[index]
        target = self.targets[index]

        # Load the image
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # 构建返回元组
        sample = (img, target)

        # 如果需要额外信息，添加到返回值中
        extra_info = {}
        if self.load_bbox:
            extra_info['bbox'] = self.bbox_info[image_id]
        if self.load_parts:
            extra_info['parts'] = self.parts_info[image_id]
        if self.load_attributes:
            extra_info['attributes'] = self.attributes_info[image_id]

        if extra_info:
            sample = sample + (extra_info,)

        return sample

    def get_images(self) -> Iterator[Image.Image]:
        """返回图像迭代器"""
        for index in range(len(self)):
            img_path = self.root / self.base_folder / 'images' / self.filtered_paths[index]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            yield image

    def get_labels(self) -> Iterator[int]:
        """返回标签迭代器"""
        for index in range(len(self)):
            yield self.filtered_labels[index] - 1  # 转换为0-based索引

    def get_images_and_labels(self) -> Iterator[Tuple[Image.Image, int]]:
        """同时返回图像和标签的迭代器"""
        for index in range(len(self)):
            img_path = self.root / self.base_folder / 'images' / self.filtered_paths[index]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.filtered_labels[index] - 1  # 转换为0-based索引
            yield image, label
