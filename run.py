# run.py
import argparse
import tomli
from pathlib import Path
import re
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import wandb
from dataclasses import dataclass
import src.models.resnet50 as resnet50
from src.data.writing_custom_datasets import CUB_200

@dataclass
class CustomDataCollator:
    def __call__(self, features):
        images = [f[0] for f in features]
        labels = [f[1] for f in features]

        images = torch.stack(images)
        labels = torch.tensor(labels)

        return {
            'pixel_values': images,
            'labels': labels
        }

class CustomModel(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.backbone = resnet50.ResNet50(num_classes=num_classes)

    def forward(self, pixel_values=None, labels=None):
        if pixel_values is None:
            raise ValueError("No input images provided")

        outputs = self.backbone(pixel_values)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

        return {
            "loss": loss,
            "logits": outputs
        }

def compute_metrics(pred):
    """计算评估指标"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {
        'accuracy': acc,
    }

def get_dataset(name, root, train_transform, test_transform):
    """获取数据集"""
    if name == "CUB-200":
        train_set = CUB_200(root=root, train=True, transform=train_transform)
        test_set = CUB_200(root=root, train=False, transform=test_transform)
        return train_set, test_set
    else:
        raise ValueError(f"Dataset {name} not supported")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with configurable dataset and training parameters')
    parser.add_argument('--dataset-config', type=str, default='configs/dataset/cub200.toml',
                        help='Path to dataset config file')
    parser.add_argument('--training-config', type=str,
                        help='Path to training config file (optional, will use latest if not specified)')
    parser.add_argument('--training-type', type=str, default='base',
                        choices=['base', 'fast'], help='Type of training config to use')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    return parser.parse_args()

def get_latest_training_config(config_type='base'):
    """
    自动获取最新的训练配置文件
    config_type: 'base' 或 'fast'
    返回最新的配置文件路径
    """
    training_config_dir = Path('configs/training')
    pattern = f"train_\\d{{8}}_{config_type}.toml"

    # 获取所有匹配的配置文件
    config_files = []
    for f in training_config_dir.glob(f'train_*_{config_type}.toml'):
        if re.match(pattern, f.name):
            date_str = f.name.split('_')[1]
            try:
                date = datetime.strptime(date_str, '%Y%m%d')
                config_files.append((date, f))
            except ValueError:
                continue

    if not config_files:
        raise FileNotFoundError(f"No training config files found for type: {config_type}")

    # 返回日期最新的配置文件
    latest_config = max(config_files, key=lambda x: x[0])[1]
    return str(latest_config)


def load_config(config_path):
    """加载TOML配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'rb') as f:
        return tomli.load(f)

def create_transform(transform_config):
    """根据配置创建数据转换pipeline"""
    transform_list = []
    for transform_name, params in transform_config:
        transform_class = getattr(transforms, transform_name)
        if params:
            transform_list.append(transform_class(**params))
        else:
            transform_list.append(transform_class())
    return transforms.Compose(transform_list)


def validate_config(config, config_type):
    """验证配置文件是否包含所有必需字段"""
    required_fields = {
        'dataset': ['name', 'root', 'num_classes', 'train_transform', 'test_transform'],
        'training': ['output_dir', 'num_train_epochs', 'learning_rate', 'per_device_train_batch_size']
    }

    for field in required_fields[config_type]:
        if field not in config[config_type]:
            raise ValueError(f"Missing required field '{field}' in {config_type} config")


def setup_wandb(wandb_config, model_name, dataset_name):
    """设置和初始化Weights & Biases"""
    wandb.init(
        project=wandb_config['project'],
        name=f"{wandb_config['run_name']}-{model_name}-{dataset_name}",
        config={
            "model": model_name,
            "dataset": dataset_name
        }
    )


def main():
    # Parse command line arguments
    args = parse_args()

    # 如果没有指定训练配置文件，则自动获取最新的
    if args.training_config is None:
        try:
            args.training_config = get_latest_training_config(args.training_type)
            print(f"Using latest training config: {args.training_config}")
        except FileNotFoundError as e:
            print(e)
            return

    # Load and validate configs
    dataset_config = load_config(args.dataset_config)
    training_config = load_config(args.training_config)

    validate_config(dataset_config, 'dataset')
    validate_config(training_config, 'training')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up wandb if enabled
    if not args.no_wandb:
        setup_wandb(
            training_config['wandb'],
            model_name="ResNet50",
            dataset_name=dataset_config['dataset']['name']
        )

    # Create transforms
    train_transform = create_transform(dataset_config['dataset']['train_transform'])
    test_transform = create_transform(dataset_config['dataset']['test_transform'])

    # Load datasets
    train_set, test_set = get_dataset(
        name=dataset_config['dataset']['name'],
        root=dataset_config['dataset']['root'],
        train_transform=train_transform,
        test_transform=test_transform
    )

    # Initialize model
    model = CustomModel(num_classes=dataset_config['dataset']['num_classes'])
    model.to(device)

    # Create training arguments
    training_args = TrainingArguments(
        **{k: v for k, v in training_config['training'].items()
           if k not in ['early_stopping']}
    )

    # Set up early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=training_config['early_stopping']['patience'],
        early_stopping_threshold=training_config['early_stopping']['threshold']
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        data_collator=CustomDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    # Train the model
    try:
        trainer.train()

        # Save the final model
        final_model_path = Path(training_config['training']['output_dir']) / 'final_model'
        trainer.save_model(str(final_model_path))
        print(f"Model saved to {final_model_path}")

        # Evaluate the model
        eval_results = trainer.evaluate()
        print("\nEvaluation Results:")
        for metric_name, value in eval_results.items():
            print(f"{metric_name}: {value:.4f}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    finally:
        # Clean up wandb
        if not args.no_wandb:
            wandb.finish()

if __name__ == '__main__':
    main()