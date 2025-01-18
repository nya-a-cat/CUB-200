import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
            transforms.Resize(256),  # Resize before crop for more consistent original size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])  # For getting the original image as tensor

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {
            'aug1': self.aug1(image),
            'aug2': self.aug2(image),
            'label': label,
            'original': self.to_tensor(image)  # Add the original image
        }

    def __len__(self):
        return len(self.base_dataset)

def create_contrastive_dataloader(dataset, aug1, aug2, batch_size=32, shuffle=True, num_workers=0):
    """
    Creates a DataLoader for contrastive learning with two augmentations per image.
    """
    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)
    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

# --- Transformation Modules ---
def get_augmentation_transforms(size=224):
    """
    Defines a set of standard augmentation transforms.
    """
    common_transforms = [
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    aug1 = transforms.Compose([
        *common_transforms,
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    ])

    aug2 = transforms.Compose([
        *common_transforms,
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        transforms.RandomRotation(30),
    ])
    return aug1, aug2

def get_inverse_transforms(size=224):
    """
    Defines inverse transforms, primarily for planar transformations.
    """
    inv_aug1 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0)
    ])

    inv_aug2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0)
    ])
    return inv_aug1, inv_aug2

# --- Visualization Modules ---
def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor to a NumPy image array for visualization.
    """
    img = tensor.permute(1, 2, 0)
    img = img.cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def display_augmented_pairs(dataloader, num_pairs=3):
    """
    Displays pairs of augmented images from a contrastive dataloader.
    """
    batch = next(iter(dataloader))
    aug1_images = batch['aug1']
    aug2_images = batch['aug2']
    labels = batch['label']

    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
    fig.suptitle('Augmented Image Pairs', fontsize=16)

    for idx in range(num_pairs):
        axes[idx, 0].imshow(tensor_to_image(aug1_images[idx]))
        axes[idx, 0].set_title(f'Aug1 - Label: {labels[idx].item()}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(tensor_to_image(aug2_images[idx]))
        axes[idx, 1].set_title(f'Aug2 - Label: {labels[idx].item()}')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_image_comparison(original_images, original_labels, aug1_images, aug2_images):
    """
    Visualizes original and augmented images side-by-side.
    """
    num_samples = original_images.size(0)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle('Original and Augmented Images', fontsize=16)

    for i in range(num_samples):
        axes[i, 0].imshow(tensor_to_image(original_images[i]))
        axes[i, 0].set_title(f'Original\nLabel: {original_labels[i].item()}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(tensor_to_image(aug1_images[i]))
        axes[i, 1].set_title(f'Aug1 (for Student)\nLabel: {original_labels[i].item()}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(tensor_to_image(aug2_images[i]))
        axes[i, 2].set_title(f'Aug2 (for Teacher)\nLabel: {original_labels[i].item()}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_feature_maps(images, features, title_prefix=''):
    """
    Visualizes feature maps for a batch of images.
    """
    num_samples = images.size(0)
    num_channels_to_visualize = min(8, features.shape[1])
    fig, axes = plt.subplots(num_channels_to_visualize, num_samples, figsize=(4 * num_samples, 4 * num_channels_to_visualize))
    fig.suptitle(f'{title_prefix} Feature Maps', fontsize=16)

    for i in range(num_samples):
        ax_img = axes[0, i]
        ax_img.imshow(tensor_to_image(images[i]))
        ax_img.set_title(f'Image {i}')
        ax_img.axis('off')
        for j in range(num_channels_to_visualize):
            ax_feat = axes[j, i]
            feature_map = features[i, j, :, :].detach().cpu().numpy()
            ax_feat.imshow(feature_map, cmap='viridis')  # Added cmap for better visualization
            ax_feat.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_feature_map_comparison(original_images, original_labels, student_features, teacher_features_aligned, title='Feature Maps Comparison'):
    """
    Visualizes and compares feature maps from the student and teacher networks for original images.
    """
    num_samples = original_images.size(0)
    num_channels_to_visualize = min(8, student_features.shape[1])
    fig, axes = plt.subplots(num_channels_to_visualize, 3 * num_samples, figsize=(12 * num_samples, 4 * num_channels_to_visualize))
    fig.suptitle(title, fontsize=16)

    for i in range(num_samples):
        ax_orig_img = axes[0, i * 3 + 0]
        ax_orig_img.imshow(tensor_to_image(original_images[i]))
        ax_orig_img.set_title(f'Original\nLabel: {original_labels[i].item()}')
        ax_orig_img.axis('off')
        for j in range(num_channels_to_visualize):
            ax_student = axes[j, i * 3 + 1]
            fs_map = student_features[i, j, :, :].detach().cpu().numpy()
            ax_student.imshow(fs_map, cmap='viridis')
            if j == 0:
                ax_student.set_title(f'Student Features')
            ax_student.axis('off')
        for j in range(num_channels_to_visualize):
            ax_teacher = axes[j, i * 3 + 2]
            ft_map = teacher_features_aligned[i, j, :, :].detach().cpu().numpy()
            ax_teacher.imshow(ft_map, cmap='viridis')
            if j == 0:
                ax_teacher.set_title(f'Teacher Features (Aligned)')
            ax_teacher.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_augmented_feature_map_comparison(aug1_images, aug2_images, student_features, teacher_features_aligned, title='Augmented Feature Maps Comparison'):
    """
    Visualizes and compares feature maps from the student and teacher networks for augmented images.
    """
    num_samples = aug1_images.size(0)
    num_channels_to_visualize = min(8, student_features.shape[1])
    fig, axes = plt.subplots(num_channels_to_visualize, 2 * num_samples, figsize=(8 * num_samples, 4 * num_channels_to_visualize))
    fig.suptitle(title, fontsize=16)
    for i in range(num_samples):
        for j in range(num_channels_to_visualize):
            ax_student_aug = axes[j, i * 2 + 0]
            fs_map_aug = student_features[i, j, :, :].detach().cpu().numpy()
            ax_student_aug.imshow(fs_map_aug, cmap='viridis')
            if j == 0:
                ax_student_aug.set_title(f'Student (Aug1)')
            ax_student_aug.axis('off')

        for j in range(num_channels_to_visualize):
            ax_teacher_aug = axes[j, i * 2 + 1]
            ft_map_aug = teacher_features_aligned[i, j, :, :].detach().cpu().numpy()
            ax_teacher_aug.imshow(ft_map_aug, cmap='viridis')
            if j == 0:
                ax_teacher_aug.set_title(f'Teacher (Aug2, Aligned)')
            ax_teacher_aug.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_consistency_loss_verification(original_images, original_labels, aug1_images, aug1_images_inv,
                                           Fs_augmented, invaug1_Fs, aug2_images, aug2_images_inv,
                                           Ft_augmented_aligned):
    """
    Visualizes the process of consistency loss verification.
    """
    num_samples = original_images.size(0)
    num_channels_to_visualize = min(8, Fs_augmented.shape[1])
    num_cols = 1 + 1 + num_channels_to_visualize + 1 + num_channels_to_visualize + 1
    fig_consistency, axes_consistency = plt.subplots(num_samples, num_cols,
                                                     figsize=(4 * num_cols, 4 * num_samples))
    fig_consistency.suptitle('Verification of Consistency Loss', fontsize=16)

    for i in range(num_samples):
        col = 0
        # Original Image
        axes_consistency[i, col].imshow(tensor_to_image(original_images[i]))
        axes_consistency[i, col].set_title(f'Original\nLabel: {original_labels[i].item()}')
        axes_consistency[i, col].axis('off')
        col += 1

        # Augmented Image 1 (Student)
        axes_consistency[i, col].imshow(tensor_to_image(aug1_images[i]))
        axes_consistency[i, col].set_title(f'Aug1 (Student)')
        axes_consistency[i, col].axis('off')
        col += 1

        # Feature Map Aug1
        for j in range(num_channels_to_visualize):
            axes_consistency[i, col].imshow(Fs_augmented[i, j, :, :].detach().cpu().numpy(), cmap='viridis')
            if j == 0:
                axes_consistency[i, col].set_title('Features Aug1')
            axes_consistency[i, col].axis('off')
            col += 1

        # Augmented Image 2 (Teacher)
        axes_consistency[i, col].imshow(tensor_to_image(aug2_images[i]))
        axes_consistency[i, col].set_title(f'Aug2 (Teacher)')
        axes_consistency[i, col].axis('off')
        col += 1

        # Feature Map Aug2
        for j in range(num_channels_to_visualize):
            axes_consistency[i, col].imshow(Ft_augmented_aligned[i, j, :, :].detach().cpu().numpy(), cmap='viridis')
            if j == 0:
                axes_consistency[i, col].set_title('Features Aug2')
            axes_consistency[i, col].axis('off')
            col += 1

    plt.tight_layout()
    plt.show()

# --- Loss Modules ---
def consistency_loss(invaug2_Ft, invaug1_Fs):
    """
    Calculates a consistency loss between two tensors after inverse augmentations.
    """
    loss = F.mse_loss(invaug2_Ft, invaug1_Fs)
    return loss

# --- Model Feature Extraction ---
def get_features(model, images, layer_name):
    """
    Extracts features from a specified layer of a given model.
    """
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
    batch_size = 4
    image_size = 224
    num_classes = 200

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms(size=image_size)

    cub_dataset_train = CUB_200(root='CUB-200', train=True, download=True)
    contrastive_dataloader = create_contrastive_dataloader(
        dataset=cub_dataset_train,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=batch_size
    )

    # --- Model Initialization ---
    student_net = models.resnet18(pretrained=True)
    teacher_net = models.resnet50(pretrained=True)
    student_net.fc = torch.nn.Linear(student_net.fc.in_features, num_classes)
    teacher_net.fc = torch.nn.Linear(teacher_net.fc.in_features, num_classes)
    for param in teacher_net.parameters():
        param.requires_grad = False

    # --- Get Batches of Data ---
    contrastive_batch = next(iter(contrastive_dataloader))
    aug1_images = contrastive_batch['aug1']
    aug2_images = contrastive_batch['aug2']
    original_images = contrastive_batch['original']
    labels = contrastive_batch['label']

    original_labels = labels

    # --- Visualize Original and Augmented Images ---
    visualize_image_comparison(original_images, original_labels, aug1_images, aug2_images)

    # --- Feature Extraction ---
    student_features_original = get_features(student_net, original_images, 'layer4')
    teacher_features_original = get_features(teacher_net, original_images, 'layer4')

    in_channels_s = student_features_original.shape[1]
    in_channels_t = teacher_features_original.shape[1]
    conv1x1_teacher = torch.nn.Conv2d(in_channels_t, in_channels_s, kernel_size=1)
    teacher_features_original_aligned = conv1x1_teacher(teacher_features_original)

    student_features_augmented = get_features(student_net, aug1_images, 'layer4')
    teacher_features_augmented = get_features(teacher_net, aug2_images, 'layer4')
    teacher_features_augmented_aligned = conv1x1_teacher(teacher_features_augmented)

    # --- Visualize Feature Maps ---
    visualize_feature_map_comparison(original_images, original_labels, student_features_original, teacher_features_original_aligned, title='Feature Maps of Original Images')
    visualize_augmented_feature_map_comparison(aug1_images, aug2_images, student_features_augmented, teacher_features_augmented_aligned, title='Feature Maps of Augmented Images')

    # --- Apply Inverse Augmentations and Get Features ---
    aug1_images_inv = torch.stack([inverse_aug1_transform(img) for img in aug1_images])
    aug2_images_inv = torch.stack([inverse_aug2_transform(img) for img in aug2_images])

    invaug1_Fs = get_features(student_net, aug1_images_inv, 'layer4')
    invaug2_Ft = conv1x1_teacher(get_features(teacher_net, aug2_images_inv, 'layer4'))

    # --- Calculate Consistency Loss ---
    loss_value = consistency_loss(invaug2_Ft, invaug1_Fs)
    print(f"Consistency Loss: {loss_value.item()}")

    # --- Visualization for Consistency Loss Verification ---
    visualize_consistency_loss_verification(original_images, original_labels, aug1_images, aug1_images_inv,
                                           student_features_augmented, invaug1_Fs, aug2_images, aug2_images_inv,
                                           teacher_features_augmented_aligned)