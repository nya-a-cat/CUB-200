# resnet50.py
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self,
                 weights: ResNet50_Weights = None,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        super().__init__()

        if pretrained and weights is None:
            weights = ResNet50_Weights.DEFAULT

        # 创建预训练的ResNet50模型
        self.model = models.resnet50(weights=weights)

        # 修改输出类别数
        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, pixel_values=None, **kwargs):
        # 兼容 Trainer 的输入
        if pixel_values is None:
            raise ValueError("No input images provided")

        # 如果有 labels，返回损失和 logits
        labels = kwargs.get('labels')
        # 计算 logits
        logits = self.model(pixel_values)

        # 如果提供了 labels，计算损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {
                'loss': loss,
                'logits': logits
            }

        return logits


def create_resnet50(num_classes=10, pretrained=True):
    model = ResNet50(
        num_classes=num_classes,
        weights=ResNet50_Weights.DEFAULT if pretrained else None
    )
    return model
