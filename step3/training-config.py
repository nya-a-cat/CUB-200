from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TrainingConfig:
    # Model parameters
    batch_size: int = 200
    image_size: int = 224
    num_classes: int = 200
    layer_name: str = 'layer4'
    unlabeled_ratio: float = 0.6
    
    # Training parameters
    epochs: int = 2
    base_lr: float = 1e-3
    
    # Loss weights and confidence parameters
    cls_weight: float = 1.0
    consistency_weight: float = 0.1
    confidence_alpha: float = 5.0
    
    # Data paths
    data_root: str = 'CUB-200'
    
    # Hardware settings
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

@dataclass
class HPSearchSpace:
    # Defines ranges for hyperparameter search
    cls_weight_range: Tuple[float, float] = (0.5, 2.0)
    consistency_weight_range: Tuple[float, float] = (0.05, 0.5)
    confidence_alpha_range: Tuple[float, float] = (1.0, 10.0)
    lr_range: Tuple[float, float] = (1e-4, 1e-2)
