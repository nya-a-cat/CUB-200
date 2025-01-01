# train.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import src.models.resnet50 as resnet50
from src.data.writing_custom_datasets import CUB_200
from dataclasses import dataclass
import wandb
from grams import Grams
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback




@dataclass
class CustomDataCollator:
    def __call__(self, features):
        images = [f[0] for f in features]
        labels = [f[1] for f in features]

        images = torch.stack(images)
        labels = torch.tensor(labels)

        return {
            'pixel_values': images,  # 修改为 pixel_values
            'labels': labels
        }


class CustomModel(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.backbone = resnet50.ResNet50(num_classes=num_classes)

    def forward(self, pixel_values=None, labels=None):  # 修改参数名
        if pixel_values is None:
            raise ValueError("No input images provided")

        outputs = self.backbone(pixel_values)  # 使用 pixel_values

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

        return {
            "loss": loss,
            "logits": outputs
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.from_numpy(logits).argmax(-1)
    accuracy = (predictions == torch.from_numpy(labels)).float().mean().item()

    return {
        "accuracy": accuracy
    }


def main():
    # 初始化 wandb
    wandb.init(
        project="CUB-200-Classification",
    )

    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载数据集
    train_set = CUB_200(
        root="../../CUB-200", train=True, download=True, transform=transform)

    test_set = CUB_200(
        root="../../CUB-200", train=False, download=True, transform=transform)

    # 训练参数
    training_args = TrainingArguments(
        output_dir='../../results',
        num_train_epochs=200,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=100,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=500,
        warmup_ratio=0.1,
        logging_dir='./logs',
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy = "epoch",
        report_to="wandb",
        run_name="ResNet50-CUB200",
        load_best_model_at_end=True,  # 训练结束时自动加载验证集表现最好的模型
        metric_for_best_model="accuracy",  # 用准确率来判断哪个是"最好的"模型
        save_total_limit=3,  # 最多保存3个检查点，超过会删除旧的
    )

    # 初始化模型
    model = CustomModel(num_classes=200)

    # 创建优化器
    optimizer = Grams(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        data_collator=CustomDataCollator(),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),  # (optimizer, scheduler) 元组
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.01  # 1% 的提升阈值
        )]
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model('./final_model')
    wandb.finish()


if __name__ == '__main__':
    main()
