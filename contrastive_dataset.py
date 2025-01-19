import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T


class ContrastiveDataset(Dataset):
    """
    Wraps a base dataset to produce pairs of differently augmented views of each image.
    Includes the original (unaugmented) image for potential use.
    """

    def __init__(self, base_dataset, augmentation1, augmentation2):
        self.base_dataset = base_dataset
        self.aug1 = augmentation1
        self.aug2 = augmentation2
        self.to_tensor = T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToDtype(torch.float32, scale=True),
            T.ToTensor(),
        ])

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {
            'aug1': self.aug1(image),
            'aug2': self.aug2(image),
            'label': label,
            'original': self.to_tensor(image),
        }

    def __len__(self):
        return len(self.base_dataset)


def create_contrastive_dataloader(dataset, aug1, aug2, batch_size=32, shuffle=True, num_workers=0, **kwargs):
    """
    Create a DataLoader for contrastive learning.

    Args:
        dataset: Base dataset to wrap
        aug1: First augmentation pipeline
        aug2: Second augmentation pipeline
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments to pass to DataLoader

    Returns:
        DataLoader: Configured DataLoader for contrastive learning
    """
    contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)
    return DataLoader(
        contrastive_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )