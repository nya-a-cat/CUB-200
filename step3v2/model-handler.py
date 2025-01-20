import torch
import torch.nn as nn
import torchvision.models as models
from config import TrainingConfig

class ModelHandler:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_models(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        # Initialize student and teacher networks
        student_net = models.resnet18(pretrained=True)
        teacher_net = models.resnet18(pretrained=True)
        
        # Freeze teacher parameters
        for param in teacher_net.parameters():
            param.requires_grad = False
            
        # Modify final layer for num_classes
        student_net.fc = nn.Linear(512, self.config.num_classes)
        teacher_net.fc = nn.Linear(512, self.config.num_classes)
        
        # Setup compression layer
        compression_layer = self.setup_compression_layer(
            teacher_net,
            student_net,
            self.config.layer_name
        )
        
        # Move to device
        student_net = student_net.to(self.device)
        teacher_net = teacher_net.to(self.device)
        compression_layer = compression_layer.to(self.device)
        
        # Set modes
        teacher_net.eval()
        student_net.train()
        
        return student_net, teacher_net, compression_layer
    
    @staticmethod
    def setup_compression_layer(
        teacher_net: nn.Module,
        student_net: nn.Module,
        layer_name: str
    ) -> nn.Module:
        """Setup 1x1 compression layer between teacher and student features."""
        teacher_dim = teacher_net._modules[layer_name][-1].conv2.out_channels
        student_dim = student_net._modules[layer_name][-1].conv2.out_channels
        return nn.Conv2d(teacher_dim, student_dim, kernel_size=1)
    
    def get_optimizer(
        self,
        student_net: nn.Module,
        compression_layer: nn.Module,
        lr: float
    ) -> torch.optim.Optimizer:
        """Setup optimizer for student and compression layer."""
        return torch.optim.Adam(
            list(student_net.parameters()) + list(compression_layer.parameters()),
            lr=lr
        )
