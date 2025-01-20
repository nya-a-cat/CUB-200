from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader
from semi_supervised_dataset import SemiSupervisedCUB200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from config import TrainingConfig

class DataHandler:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.aug1_transform, self.aug2_transform = get_augmentation_transforms(
            size=config.image_size
        )
        self.inverse_aug1_transform, self.inverse_aug2_transform = get_inverse_transforms()
        
    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        # Training data with contrastive augmentations
        train_dataset = SemiSupervisedCUB200(
            root=self.config.data_root,
            train=True,
            transform=transforms.ToDtype,
            unlabeled_ratio=self.config.unlabeled_ratio
        )
        
        train_loader = create_contrastive_dataloader(
            dataset=train_dataset,
            aug1=self.aug1_transform,
            aug2=self.aug2_transform,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )
        
        # Test data
        test_dataset = SemiSupervisedCUB200(
            root=self.config.data_root,
            train=False,
            transform=transforms.ToDtype,
            unlabeled_ratio=0.0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,  # Fixed test batch size
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )
        
        return train_loader, test_loader
    
    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Process a contrastive batch and move to device."""
        return {
            'original': batch['original'].to(device),
            'aug1': batch['aug1'].to(device),
            'aug2': batch['aug2'].to(device),
            'label': batch['label'].to(device)
        }
        
    def inverse_transform_features(
        self,
        features_s: torch.Tensor,
        features_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply inverse transforms to student and teacher features."""
        return (
            self.inverse_aug1_transform(features_s),
            self.inverse_aug2_transform(features_t)
        )
