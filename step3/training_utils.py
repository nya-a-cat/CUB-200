import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader
import wandb
from config import TrainingConfig

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

class LossComputer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def compute_classification_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        confidence_weights: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = self.criterion(student_logits, labels)
        weighted_ce_loss = (ce_loss * confidence_weights).mean()
        return self.config.cls_weight * weighted_ce_loss
        
    def compute_consistency_loss(
        self,
        teacher_features: torch.Tensor,
        student_features: torch.Tensor
    ) -> torch.Tensor:
        return self.config.consistency_weight * F.mse_loss(
            teacher_features,
            student_features
        )
        
    def compute_confidence_weights(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor
    ) -> torch.Tensor:
        diff = (p1 - p2).pow(2).sum(dim=1).sqrt()
        return torch.exp(-self.config.confidence_alpha * diff)

def train_epoch(
    student_net: nn.Module,
    teacher_net: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_computer: LossComputer,
    device: torch.device,
    compression_layer: Optional[nn.Module] = None
) -> Dict[str, float]:
    student_net.train()
    teacher_net.eval()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_consistency_loss = 0.0
    
    for batch in train_loader:
        # Training logic from your main.py
        # ... (implement the training loop from your original code)
        pass
        
    metrics = {
        'train_loss': total_loss / len(train_loader),
        'train_cls_loss': total_cls_loss / len(train_loader),
        'train_consistency_loss': total_consistency_loss / len(train_loader)
    }
    
    return metrics

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    
    return accuracy, avg_loss
