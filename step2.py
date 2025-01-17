import torch.utils.data
import torchvision.models as models
from writing_custom_datasets import CUB_200
import torch.utils.data
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

def create_contrastive_dataloader(dataset, aug1, aug2, batch_size=32, shuffle=True, num_workers=0):
    class ContrastiveDataset(Dataset):
        def __init__(self, base_dataset, augmentation1, augmentation2):
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
    batch = next(iter(dataloader))
    aug1_images = batch['aug1']
    aug2_images = batch['aug2']
    labels = batch['label']

    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
    fig.suptitle('Augmented Image Pairs from CUB-200 Dataset', fontsize=16)

    def tensor_to_img(tensor):
        img = tensor.permute(1, 2, 0)
        img = img.numpy().clip(0, 1)
        return img

    for idx in range(num_pairs):
        axes[idx, 0].imshow(tensor_to_img(aug1_images[idx]))
        axes[idx, 0].set_title(f'Aug1 - Label: {labels[idx].item()}')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(tensor_to_img(aug2_images[idx]))
        axes[idx, 1].set_title(f'Aug2 - Label: {labels[idx].item()}')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

def tensor_to_img_np(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Define transforms for augmented views
    aug1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])

    aug2 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((256, 256))
    ])

    # Create DataLoaders
    dataloader = create_contrastive_dataloader(
        dataset=CUB_200(root='CUB-200', train=True, download=True),
        aug1=aug1,
        aug2=aug2,
        batch_size=4
    )
    original_dataset = CUB_200(root='CUB-200', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]))
    original_dataloader = DataLoader(original_dataset, batch_size=4, shuffle=True)

    # Get batches of data
    batch = next(iter(dataloader))
    aug1_images = batch['aug1']
    aug2_images = batch['aug2']
    labels = batch['label']
    original_batch = next(iter(original_dataloader))
    original_images = original_batch[0]
    original_labels = original_batch[1]

    # Visualize original and augmented images
    num_samples = original_images.size(0)
    fig_all_images, axes_all_images = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig_all_images.suptitle('Original and Augmented Images', fontsize=16)

    for i in range(num_samples):
        axes_all_images[i, 0].imshow(tensor_to_img_np(original_images[i]))
        axes_all_images[i, 0].set_title(f'Original\nLabel: {original_labels[i].item()}')
        axes_all_images[i, 0].axis('off')

        axes_all_images[i, 1].imshow(tensor_to_img_np(aug1_images[i]))
        axes_all_images[i, 1].set_title(f'Aug1 (for Student)\nLabel: {labels[i].item()}')
        axes_all_images[i, 1].axis('off')

        axes_all_images[i, 2].imshow(tensor_to_img_np(aug2_images[i]))
        axes_all_images[i, 2].set_title(f'Aug2 (for Teacher)\nLabel: {labels[i].item()}')
        axes_all_images[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

    StudentNet = models.resnet18(pretrained=True)
    TeacherNet = models.resnet50(pretrained=True)
    num_classes = 200
    StudentNet.fc = torch.nn.Linear(StudentNet.fc.in_features, num_classes)
    TeacherNet.fc = torch.nn.Linear(TeacherNet.fc.in_features, num_classes)
    for param in TeacherNet.parameters():
        param.requires_grad = False

    features_student = {}
    features_teacher = {}
    def get_features_student(name):
        def hook(model, input, output):
            features_student[name] = output.detach()
        return hook
    def get_features_teacher(name):
        def hook(model, input, output):
            features_teacher[name] = output.detach()
        return hook

    StudentNet.layer4.register_forward_hook(get_features_student('student_features'))
    TeacherNet.layer4.register_forward_hook(get_features_teacher('teacher_features'))

    # Feature maps for original images
    _ = StudentNet(original_images)
    Fs_original = features_student['student_features']
    in_channels_s = Fs_original.shape[1]
    _ = TeacherNet(original_images)
    Ft_original = features_teacher['teacher_features']
    in_channels_t = Ft_original.shape[1]
    conv1x1_teacher = torch.nn.Conv2d(in_channels_t, in_channels_s, kernel_size=1)
    Ft_original_aligned = conv1x1_teacher(Ft_original)

    # Feature maps for augmented images
    _ = StudentNet(aug1_images)
    Fs_augmented = features_student['student_features']
    _ = TeacherNet(aug2_images)
    Ft_augmented = features_teacher['teacher_features']
    Ft_augmented_aligned = conv1x1_teacher(Ft_augmented)

    # Visualize feature maps for original images (vertically arranged)
    num_channels_to_visualize = min(8, Fs_original.shape[1])
    fig_orig_features, axes_orig_features = plt.subplots(num_channels_to_visualize, 3 * num_samples, figsize=(12 * num_samples, 4 * num_channels_to_visualize))
    fig_orig_features.suptitle('Feature Maps of Original Images', fontsize=16)
    for i in range(num_samples):
        ax_orig_img = axes_orig_features[0, i * 3 + 0]
        ax_orig_img.imshow(tensor_to_img_np(original_images[i]))
        ax_orig_img.set_title(f'Original\nLabel: {original_labels[i].item()}')
        ax_orig_img.axis('off')
        for j in range(num_channels_to_visualize):
            ax_student = axes_orig_features[j, i * 3 + 1]
            fs_map = Fs_original[i, j, :, :].detach().cpu().numpy()
            ax_student.imshow(fs_map, cmap='viridis')
            if j == 0:
                ax_student.set_title(f'Student Features')
            ax_student.axis('off')
        for j in range(num_channels_to_visualize):
            ax_teacher = axes_orig_features[j, i * 3 + 2]
            ft_map = Ft_original_aligned[i, j, :, :].detach().cpu().numpy()
            ax_teacher.imshow(ft_map, cmap='viridis')
            if j == 0:
                ax_teacher.set_title(f'Teacher Features (Aligned)')
            ax_teacher.axis('off')
    plt.tight_layout()
    plt.show()

    # Visualize feature maps for augmented images (vertically arranged)
    fig_aug_features, axes_aug_features = plt.subplots(num_channels_to_visualize, 2 * num_samples, figsize=(8 * num_samples, 4 * num_channels_to_visualize))
    fig_aug_features.suptitle('Feature Maps of Augmented Images', fontsize=16)
    for i in range(num_samples):
        # StudentNet features on augmented images
        for j in range(num_channels_to_visualize):
            ax_student_aug = axes_aug_features[j, i * 2 + 0]
            fs_map_aug = Fs_augmented[i, j, :, :].detach().cpu().numpy()
            ax_student_aug.imshow(fs_map_aug, cmap='viridis')
            if j == 0:
                ax_student_aug.set_title(f'Student (Aug1)')
            ax_student_aug.axis('off')

        # TeacherNet features on augmented images (aligned)
        for j in range(num_channels_to_visualize):
            ax_teacher_aug = axes_aug_features[j, i * 2 + 1]
            ft_map_aug = Ft_augmented_aligned[i, j, :, :].detach().cpu().numpy()
            ax_teacher_aug.imshow(ft_map_aug, cmap='viridis')
            if j == 0:
                ax_teacher_aug.set_title(f'Teacher (Aug2, Aligned)')
            ax_teacher_aug.axis('off')
    plt.tight_layout()
    plt.show()