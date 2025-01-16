# step2.py
import torch.utils.data

import torch.utils.data

import torchvision.models as models
from writing_custom_datasets import CUB_200
import torch.utils.data
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 设置全局字体大小
plt.rcParams.update({'font.size': 30})  # 可以尝试 12, 14, 16 或更大的值
# 全局设置标题字体大小
plt.rcParams['axes.titlesize'] = 30  # 你可以尝试 16, 18, 20 或更大的值

def create_contrastive_dataloader(dataset,aug1,aug2,batch_size=32,shuffle=True,num_workers=0
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

    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)

    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def display_comparison_of_augmented_pairs(dataloader, num_pairs=3):
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

def inverse_transform(transformations):
    """
    Create an inverse of the given transformations for spatial or planar reversal.

    Args:
        transformations: List of torchvision transformations to reverse.

    Returns:
        torchvision.transforms.Compose object that reverses the given transformations.
    """
    reversed_transforms = []

    for transform in reversed(transformations.transforms):  # Reverse the order of transformations
        if isinstance(transform, transforms.Normalize):
            # To reverse Normalize, create an un-normalization transform
            mean = torch.tensor(transform.mean)
            std = torch.tensor(transform.std)
            reversed_transforms.append(transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()))
        elif isinstance(transform, transforms.RandomHorizontalFlip):
            # Reverse Horizontal Flip by adding it again
            reversed_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
        elif isinstance(transform, transforms.RandomVerticalFlip):
            # Reverse Vertical Flip by adding it again
            reversed_transforms.append(transforms.RandomVerticalFlip(p=1.0))
        elif isinstance(transform, transforms.ToTensor):
            # Tensor conversion can be seen as a no-op (no need for reversal)
            continue
        else:
            # Skip transformations that cannot be inverted
            continue

    # Return the reversed transformations as a Compose object
    return transforms.Compose(reversed_transforms)



def create_contrastive_dataloader(dataset,aug1,aug2,batch_size=32,shuffle=True,num_workers=0
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

    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)

    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def display_comparison_of_augmented_pairs(dataloader, num_pairs=3):
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

def inverse_transform(transformations):
    """
    Create an inverse of the given transformations for spatial or planar reversal.

    Args:
        transformations: List of torchvision transformations to reverse.

    Returns:
        torchvision.transforms.Compose object that reverses the given transformations.
    """
    reversed_transforms = []

    for transform in reversed(transformations.transforms):  # Reverse the order of transformations
        if isinstance(transform, transforms.Normalize):
            # To reverse Normalize, create an un-normalization transform
            mean = torch.tensor(transform.mean)
            std = torch.tensor(transform.std)
            reversed_transforms.append(transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()))
        elif isinstance(transform, transforms.RandomHorizontalFlip):
            # Reverse Horizontal Flip by adding it again
            reversed_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
        elif isinstance(transform, transforms.RandomVerticalFlip):
            # Reverse Vertical Flip by adding it again
            reversed_transforms.append(transforms.RandomVerticalFlip(p=1.0))
        elif isinstance(transform, transforms.ToTensor):
            # Tensor conversion can be seen as a no-op (no need for reversal)
            continue
        elif isinstance(transform, transforms.RandomResizedCrop):
            # Cannot directly reverse RandomResizedCrop deterministically
            print("Warning: Cannot perfectly reverse RandomResizedCrop.")
            continue
        elif isinstance(transform, transforms.ColorJitter):
            print("Warning: Cannot perfectly reverse ColorJitter.")
            continue
        elif isinstance(transform, transforms.RandomRotation):
            print("Warning: Cannot perfectly reverse RandomRotation.")
            continue
        elif isinstance(transform, transforms.RandomAffine):
            print("Warning: Cannot perfectly reverse RandomAffine.")
            continue
        else:
            # Skip transformations that cannot be inverted
            continue

    # Return the reversed transformations as a Compose object
    return transforms.Compose(reversed_transforms)

def create_contrastive_dataloader(dataset, aug1, aug2, batch_size=32, shuffle=True, num_workers=0):
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

    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)

    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def display_comparison_of_augmented_pairs(dataloader, num_pairs=3):
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

def inverse_transform(transformations):
    """
    Create an inverse of the given transformations for spatial or planar reversal.

    Args:
        transformations: List of torchvision transformations to reverse.

    Returns:
        torchvision.transforms.Compose object that reverses the given transformations.
    """
    reversed_transforms = []

    for transform in reversed(transformations.transforms):  # Reverse the order of transformations
        if isinstance(transform, transforms.Normalize):
            # To reverse Normalize, create an un-normalization transform
            mean = torch.tensor(transform.mean)
            std = torch.tensor(transform.std)
            reversed_transforms.append(transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()))
        elif isinstance(transform, transforms.RandomHorizontalFlip):
            # Reverse Horizontal Flip by adding it again
            reversed_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
        elif isinstance(transform, transforms.RandomVerticalFlip):
            # Reverse Vertical Flip by adding it again
            reversed_transforms.append(transforms.RandomVerticalFlip(p=1.0))
        elif isinstance(transform, transforms.ToTensor):
            # Tensor conversion can be seen as a no-op (no need for reversal)
            continue
        elif isinstance(transform, transforms.RandomResizedCrop):
            # Cannot directly reverse RandomResizedCrop deterministically
            print("Warning: Cannot perfectly reverse RandomResizedCrop.")
            continue
        elif isinstance(transform, transforms.ColorJitter):
            print("Warning: Cannot perfectly reverse ColorJitter.")
            continue
        elif isinstance(transform, transforms.RandomRotation):
            print("Warning: Cannot perfectly reverse RandomRotation.")
            continue
        elif isinstance(transform, transforms.RandomAffine):
            print("Warning: Cannot perfectly reverse RandomAffine.")
            continue
        else:
            # Skip transformations that cannot be inverted
            continue

    # Return the reversed transformations as a Compose object
    return transforms.Compose(reversed_transforms)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    aug1 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    aug2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader = create_contrastive_dataloader(
        dataset=CUB_200(root='CUB-200', train=True, download=True),
        aug1=aug1,
        aug2=aug2,
        batch_size=4  # 减小 batch size 以方便可视化
    )

    StudentNet = models.resnet18(pretrained=True)
    TeacherNet = models.resnet50(pretrained=True)
    num_classes = 200
    StudentNet.fc = torch.nn.Linear(StudentNet.fc.in_features, num_classes)
    TeacherNet.fc = torch.nn.Linear(TeacherNet.fc.in_features, num_classes)

    # 2.5) 对TeacherNet的参数进行冻结，不进行更新。
    for param in TeacherNet.parameters():
        param.requires_grad = False

    # 定义分类损失函数
    classification_criterion = torch.nn.CrossEntropyLoss()

    # 注册 hook 来获取特征图
    features_student = {}
    features_teacher = {}
    def get_features_student(name):
        def hook(model, input, output):
            features_student[name] = output
        return hook

    def get_features_teacher(name):
        def hook(model, input, output):
            features_teacher[name] = output
        return hook

    StudentNet.layer4.register_forward_hook(get_features_student('student_features'))
    TeacherNet.layer4.register_forward_hook(get_features_teacher('teacher_features'))

    # 获取一批数据
    batch = next(iter(dataloader))
    aug_1_image = batch['aug1']
    aug_2_image = batch['aug2']
    labels = batch['label']

    # 前向传播
    student_output = StudentNet(aug_1_image)
    _ = TeacherNet(aug_2_image)
    Fs = features_student['student_features']
    Ft = features_teacher['teacher_features']

    # 2.5) 对StudentNet额外有分类代价函数。
    classification_loss = classification_criterion(student_output, labels)
    print(f"Classification Loss: {classification_loss.item()}")

    # 使用1x1卷积调整通道数
    conv1x1 = torch.nn.Conv2d(Ft.size(1), Fs.size(1), kernel_size=1).to(aug_1_image.device)
    Ft = conv1x1(Ft)

    # 计算一致性损失 (例如，使用均方误差)
    consistency_loss = F.mse_loss(Fs, Ft)
    print(f"Consistency Loss: {consistency_loss.item()}")

    # 可视化 Fs, Ft 和它们之间的差异

    # 选择要可视化的样本和通道 (这里选择第一个样本的前 8 个通道)
    sample_index = 0
    num_channels_to_visualize = min(8, Fs.shape[1])

    fig, axes = plt.subplots(3, num_channels_to_visualize, figsize=(16, 8))
    fig.suptitle('Visualization of Fs, Ft, and their Difference', fontsize=16)

    for i in range(num_channels_to_visualize):
        # 可视化 Fs
        ax_fs = axes[0, i]
        fs_map = Fs[sample_index, i, :, :].detach().cpu().numpy()
        ax_fs.imshow(fs_map, cmap='viridis')
        ax_fs.set_title(f'Fs Channel {i}')
        ax_fs.axis('off')

        # 可视化 Ft
        ax_ft = axes[1, i]
        ft_map = Ft[sample_index, i, :, :].detach().cpu().numpy()
        ax_ft.imshow(ft_map, cmap='viridis')
        ax_ft.set_title(f'Ft Channel {i}')
        ax_ft.axis('off')

        # 可视化差异 |Fs - Ft|
        ax_diff = axes[2, i]
        diff_map = torch.abs(Fs[sample_index, i, :] - Ft[sample_index, i, :]).detach().cpu().numpy()
        ax_diff.imshow(diff_map, cmap='viridis')
        ax_diff.set_title(f'|Fs - Ft| Channel {i}')
        ax_diff.axis('off')

    plt.tight_layout()
    plt.show()

    # 如果想更贴近 MSE Loss 的计算，可以可视化差异的平方
    squared_diff = (Fs - Ft) ** 2
    fig_squared_diff, axes_squared_diff = plt.subplots(1, num_channels_to_visualize, figsize=(16, 4))
    fig_squared_diff.suptitle('Visualization of (Fs - Ft)^2', fontsize=16)

    for i in range(num_channels_to_visualize):
        ax = axes_squared_diff[i]
        squared_diff_map = squared_diff[sample_index, i, :, :].detach().cpu().numpy()
        ax.imshow(squared_diff_map, cmap='viridis')
        ax.set_title(f'Squared Diff Channel {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 2.6) 设计可视化证明你的consistency loss没有施加错误，例如图片变换和特征逆变换都是位置对应的，平面变换没有错误。
    def visualize_consistency(image, transform, model_student, model_teacher, feature_layer_student, feature_layer_teacher):
        """
        Visualizes the consistency of feature maps after applying a transformation.

        Args:
            image: A single input image tensor.
            transform: The transformation to apply.
            model_student: The StudentNet model.
            model_teacher: The TeacherNet model.
            feature_layer_student: The name of the feature layer in StudentNet.
            feature_layer_teacher: The name of the feature layer in TeacherNet.
        """
        model_student.eval()
        model_teacher.eval()

        # Register hooks to get feature maps
        features_before_student = {}
        features_after_student = {}
        features_before_teacher = {}
        features_after_teacher = {}

        def hook_before_student(module, input, output):
            features_before_student['features'] = output.detach()
        def hook_after_student(module, input, output):
            features_after_student['features'] = output.detach()
        def hook_before_teacher(module, input, output):
            features_before_teacher['features'] = output.detach()
        def hook_after_teacher(module, input, output):
            features_after_teacher['features'] = output.detach()

        handle_before_student = feature_layer_student.register_forward_hook(hook_before_student)
        handle_after_student = feature_layer_student.register_forward_hook(hook_after_student)
        handle_before_teacher = feature_layer_teacher.register_forward_hook(hook_before_teacher)
        handle_after_teacher = feature_layer_teacher.register_forward_hook(hook_after_teacher)

        # Get feature maps before transformation
        _ = model_student(image.unsqueeze(0))
        features_before_student_map = features_before_student['features'][0]
        _ = model_teacher(image.unsqueeze(0))
        features_before_teacher_map = features_before_teacher['features'][0]

        # Apply transformation
        transformed_image = transform(image).unsqueeze(0)

        # Get feature maps after transformation
        _ = model_student(transformed_image)
        features_after_student_map = features_after_student['features'][0]
        _ = model_teacher(transformed_image)
        features_after_teacher_map = features_after_teacher['features'][0]

        # Remove hooks
        handle_before_student.remove()
        handle_after_student.remove()
        handle_before_teacher.remove()
        handle_after_teacher.remove()

        # Visualize
        num_channels = min(8, features_before_student_map.shape[0])
        fig, axes = plt.subplots(4, num_channels, figsize=(16, 12))
        fig.suptitle(f'Consistency Visualization with Transformation: {transform}', fontsize=16)

        def tensor_to_img_np(tensor):
            img = tensor.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize for visualization
            return img

        axes[0, 0].imshow(tensor_to_img_np(image))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        for i in range(1, num_channels):
            axes[0, i].axis('off')

        axes[1, 0].imshow(tensor_to_img_np(transformed_image.squeeze(0)))
        axes[1, 0].set_title('Transformed Image')
        axes[1, 0].axis('off')
        for i in range(1, num_channels):
            axes[1, i].axis('off')

        for i in range(num_channels):
            # StudentNet
            axes[2, i].imshow(features_before_student_map[i].cpu().numpy(), cmap='viridis')
            axes[2, i].set_title(f'Student Before Ch {i}')
            axes[2, i].axis('off')

            axes[3, i].imshow(features_after_student_map[i].cpu().numpy(), cmap='viridis')
            axes[3, i].set_title(f'Student After Ch {i}')
            axes[3, i].axis('off')

        plt.tight_layout()
        plt.show()

        # Visualize TeacherNet features (similar to StudentNet) - Optional but good for comparison
        fig_teacher, axes_teacher = plt.subplots(2, num_channels, figsize=(16, 8))
        fig_teacher.suptitle(f'TeacherNet Feature Consistency with Transformation: {transform}', fontsize=16)
        for i in range(num_channels):
            axes_teacher[0, i].imshow(features_before_teacher_map[i].cpu().numpy(), cmap='viridis')
            axes_teacher[0, i].set_title(f'Teacher Before Ch {i}')
            axes_teacher[0, i].axis('off')

            axes_teacher[1, i].imshow(features_after_teacher_map[i].cpu().numpy(), cmap='viridis')
            axes_teacher[1, i].set_title(f'Teacher After Ch {i}')
            axes_teacher[1, i].axis('off')
        plt.tight_layout()
        plt.show()

    # 选择一个样本进行可视化
    sample_data = next(iter(dataloader))
    sample_image = sample_data['aug1'][0] # 使用 aug1 的图片
    original_image_unnormalized = inverse_transform(aug1)(sample_image.cpu()).clamp(0, 1) # 反normalize回来方便查看

    # 定义一些用于测试的变换 (需要是 transforms 对象)
    horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
    vertical_flip = transforms.RandomVerticalFlip(p=1.0)
    translate_transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0)) # 水平平移

    # 获取 StudentNet 和 TeacherNet 的目标特征层
    student_layer4 = StudentNet.layer4
    teacher_layer4 = TeacherNet.layer4

    # 进行可视化
    visualize_consistency(original_image_unnormalized, horizontal_flip, StudentNet, TeacherNet, student_layer4, teacher_layer4)
    visualize_consistency(original_image_unnormalized, vertical_flip, StudentNet, TeacherNet, student_layer4, teacher_layer4)
    visualize_consistency(original_image_unnormalized, translate_transform, StudentNet, TeacherNet, student_layer4, teacher_layer4)