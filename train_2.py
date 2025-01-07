import torchvision.models as models
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb  # 可选：用于实验追踪

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据增强和预处理
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

# 验证函数
def validate(model, test_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    valid_total = 0
    valid_correct = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            valid_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            valid_total += batch_labels.size(0)
            valid_correct += (predicted == batch_labels).sum().item()

    avg_valid_loss = valid_loss / len(test_loader)
    valid_accuracy = 100 * valid_correct / valid_total

    return avg_valid_loss, valid_accuracy

#for every picture, this dataloader return aug1(pic) and aug2(pic)
class ContrastiveDataset(Dataset):
    """
    Wrapper dataset that applies two different augmentations to each image
    for contrastive learning approaches.
    """

    def __init__(self, base_dataset, augmentation1, augmentation2):
        """
        Args:
            base_dataset: Original dataset containing images
            augmentation1: First augmentation transform
            augmentation2: Second augmentation transform
        """
        self.base_dataset = base_dataset
        self.aug1 = augmentation1
        self.aug2 = augmentation2

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {
            'aug1': self.aug1(image),
            'aug2': self.aug2(image),
            'label': label
        }

    def __len__(self):
        return len(self.base_dataset)

def create_contrastive_dataloader(
        dataset,
        aug1,
        aug2,
        batch_size=32,
        shuffle=True,
        num_workers=0
):
    """
    Creates a DataLoader that returns pairs of differently augmented views of the same image.

    Args:
        dataset: Base dataset containing images
        aug1: First augmentation transform
        aug2: Second augmentation transform
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader that yields dictionaries containing:
            - 'aug1': First augmented view
            - 'aug2': Second augmented view
            - 'label': Original label
    """
    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)

    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

# 其中aug2的增强范围要更大
aug1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
])

aug2 = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # 调整裁剪的范围，更大的裁剪变换
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),  # 扩大色调变化范围
    transforms.RandomRotation(30),  # 加入旋转
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # 平移变换，增大范围
    transforms.ToTensor(),
])

# Create the dataloader
dataloader = create_contrastive_dataloader(
    dataset=CUB_200(root='CUB-200', train=True, download=True),
    aug1=aug1,
    aug2=aug2,
    batch_size=32
)

# use plt to print aug1 aug2 and label for valid this dataloader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def show_augmented_pairs(dataloader, num_pairs=3):
    """
    Display pairs of augmented images from the contrastive dataloader

    Args:
        dataloader: Contrastive dataloader returning aug1, aug2 pairs
        num_pairs: Number of image pairs to display
    """
    # Get a batch of data
    batch = next(iter(dataloader))
    aug1_images = batch['aug1']
    aug2_images = batch['aug2']
    labels = batch['label']

    # Create a figure with subplots for each pair
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
    fig.suptitle('Augmented Image Pairs from CUB-200 Dataset', fontsize=16)

    # Helper function to convert tensor to displayable image
    def tensor_to_img(tensor):
        # Remove batch dimension and move channels to last dimension
        img = tensor.permute(1, 2, 0)
        # Convert to numpy and clip to valid range
        img = img.numpy().clip(0, 1)
        return img

    for idx in range(num_pairs):
        # Display first augmentation
        axes[idx, 0].imshow(tensor_to_img(aug1_images[idx]))
        axes[idx, 0].set_title(f'Aug1 - Label: {labels[idx].item()}')
        axes[idx, 0].axis('off')

        # Display second augmentation
        axes[idx, 1].imshow(tensor_to_img(aug2_images[idx]))
        axes[idx, 1].set_title(f'Aug2 - Label: {labels[idx].item()}')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Visualize some pairs

show_augmented_pairs(dataloader)

# 2.2）把aug1(x)输入StudentNet，其中最后一层feature map记为Fs。
# 2.3）把aug2(x)输入TeacherNet，
# 其中最后一层feature map记为Ft。

batch = next(iter(dataloader))
aug1_images = batch['aug1']
StudentNet = models.resnet18(pretrained=True)
StudentNet.fc = torch.nn.Linear(StudentNet.fc.in_features, 200)

aug2_images = batch['aug2']
TeacherNet = models.resnet50(pretrained=True)
TeacherNet.fc = torch.nn.Linear(TeacherNet.fc.in_features, 200)

Fs = StudentNet.fc
Ft = TeacherNet.fc

# 使用Conv1x1压缩通道数，并确保学生和教师网络的通道数一致
align_conv = (
    torch.nn.Conv2d
    (Fs.in_features,
     Ft.in_features,
     kernel_size=1,
     stride=1,
     padding=0,
     bias=False))

# 2.4)
# L= || invaug2(Ft) - invaug1(Fs) ||.
# Compute inverse augmentations and consistency loss
# Helper function to invert common transforms

def compute_consistency_loss(
        features_s, features_t, transform1, transform2):
    """
    Compute consistency loss between student and teacher features after inverse transforms

    Args:
        features_s: Student network features
        features_t: Teacher network features
        transform1: First transformation (torchvision.transforms)
        transform2: Second transformation (torchvision.transforms)
    """

    # Get inverse transforms
    inv_transform1 = invert_transform(transform1)
    inv_transform2 = invert_transform(transform2)

    # Apply inverse transforms to features
    inv_features_s = inv_transform1(features_s)
    inv_features_t = inv_transform2(features_t)

    # Compute MSE loss between inversely transformed features
    consistency_loss = F.mse_loss(inv_features_s, inv_features_t)

    return consistency_loss

# Freeze teacher network parameters
for param in TeacherNet.parameters():
    param.requires_grad = False

# TODO:total_loss = cls_loss + consistency_loss
# Add any other losses you have

# 2.6）设计可视化证明
# consistency loss没有错误，

# def f(batch, arg1, arg2, criterion, )
#     use plt to show origin image, arg1image, arg2image,
#     and inversarg1 image ,inversarg2 image,
#     we have def compute_consistency_loss
#     return consistency_loss and def invert_transform
#     return transform
#     计算consistency loss并显示在图片上

def invert_transform(compose_transform):
    """
    Generate inverse transformations for a given transforms.Compose object.
    Note: Some transformations like random augmentations cannot be perfectly inverted
    due to their stochastic nature.

    Args:
        compose_transform (transforms.Compose): Original composition of transforms

    Returns:
        transforms.Compose: Composition of approximate inverse transforms
    """
    inverse_transforms = []

    for transform in reversed(compose_transform.transforms):
        if isinstance(transform, transforms.RandomResizedCrop):
            # RandomResizedCrop 不能完美还原，但可以resize到原始大小
            inverse_transforms.append(transforms.Resize((256, 256)))  # 假设原始大小

        elif isinstance(transform, transforms.RandomHorizontalFlip):
            # 随机翻转无法准确还原，因为不知道是否进行了翻转
            # 保持原样即可，不添加逆变换
            pass

        elif isinstance(transform, transforms.ColorJitter):
            # 颜色抖动的逆变换应该使用相反的参数
            inverse_transforms.append(transforms.ColorJitter(
                brightness=transform.brightness if transform.brightness else None,
                contrast=transform.contrast if transform.contrast else None,
                saturation=transform.saturation if transform.saturation else None,
                hue=(-transform.hue, transform.hue) if transform.hue else None
            ))

        elif isinstance(transform, transforms.RandomRotation):
            # 随机旋转的逆变换需要知道具体旋转了多少度
            # 这里保存旋转角度作为transform的属性
            if hasattr(transform, 'actual_angle'):
                inverse_transforms.append(transforms.RandomRotation(
                    degrees=-transform.actual_angle
                ))

        elif isinstance(transform, transforms.RandomAffine):
            # 仿射变换的逆变换需要原始变换矩阵的逆矩阵
            # 这里需要在transform时保存变换矩阵
            if hasattr(transform, 'last_matrix'):
                # 使用numpy.linalg.inv计算逆矩阵
                import numpy as np
                inv_matrix = np.linalg.inv(transform.last_matrix)
                inverse_transforms.append(
                    transforms.Lambda(lambda x: F.affine(
                        x,
                        angle=inv_matrix[0, 0],
                        translate=(inv_matrix[0, 2], inv_matrix[1, 2]),
                        scale=1.0 / np.sqrt(inv_matrix[0, 0] ** 2 + inv_matrix[0, 1] ** 2),
                        shear=np.arctan2(inv_matrix[0, 1], inv_matrix[0, 0])
                    ))
                )

        elif isinstance(transform, transforms.ToTensor):
            inverse_transforms.append(transforms.ToPILImage())

        elif isinstance(transform, transforms.Normalize):
            # Normalize的逆变换需要用相反的mean和std
            mean = torch.tensor(transform.mean)
            std = torch.tensor(transform.std)
            inverse_transforms.append(transforms.Normalize(
                mean=(-mean / std),
                std=(1.0 / std)
            ))

    return transforms.Compose(inverse_transforms)

def visualize_consistency_loss(batch, arg1, arg2, criterion):
    """
    计算一致性损失并显示原始图像、增强图像和反向增强图像。

    该函数使用给定的损失函数计算一致性损失，但仅进行可视化，不返回损失值。

    Args:
        batch: 包含图像和标签的一个批次
        arg1: 第一个增强变换
        arg2: 第二个增强变换
        criterion: 损失函数
        device: 计算设备（如 'cuda' 或 'cpu'）
    """
    # 获取增强后的图像（假设 batch 中的 aug1 和 aug2 已经是增强过的图片）
    aug1_images = batch['aug1']
    aug2_images = batch['aug2']

    # 假设 StudentNet 和 TeacherNet 是预定义好的网络
    StudentNet.eval()
    TeacherNet.eval()

    with torch.no_grad():
        # 计算学生和教师网络的特征
        features_s = StudentNet(aug1_images)
        features_t = TeacherNet(aug2_images)

        # 使用 criterion 计算一致性损失
        consistency_loss = criterion(features_s, features_t, arg1, arg2)

    # 可视化原始图像和增强图像
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Consistency Loss: {consistency_loss.item():.4f}", fontsize=16)

    # 展示原始图像
    axes[0, 0].imshow(batch['aug1'][0].cpu().numpy().transpose(1, 2, 0))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(batch['aug1'][1].cpu().numpy().transpose(1, 2, 0))
    axes[0, 1].set_title("Aug1 Image")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(batch['aug2'][0].cpu().numpy().transpose(1, 2, 0))
    axes[0, 2].set_title("Aug2 Image")
    axes[0, 2].axis('off')

    # 可视化反向增强后的图像
    inv_aug1 = invert_transform(arg1)(aug1_images).cpu().numpy().transpose(0, 2, 3, 1)[0]
    inv_aug2 = invert_transform(arg2)(aug2_images).cpu().numpy().transpose(0, 2, 3, 1)[0]

    axes[1, 0].imshow(inv_aug1)
    axes[1, 0].set_title("Inverse Aug1 Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(inv_aug2)
    axes[1, 1].set_title("Inverse Aug2 Image")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

visualize_consistency_loss(next(iter(dataloader)), aug1, aug2, compute_consistency_loss)