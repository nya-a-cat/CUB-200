import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import os
import wandb
import argparse
import configparser

# --- 1. 配置管理 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# --- 2. 数据加载器模块 ---
class DatasetLoader:
    def __init__(self, config):
        self.config = config

    def get_augmentation_transforms(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.config.getint("Training", "image_size"), scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_inverse_transforms(self):
        return transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

    def load_train_data(self):
        from semi_supervised_dataset import SemiSupervisedCUB200
        return SemiSupervisedCUB200(
            root='CUB-200',
            train=True,
            transform=transforms.ToTensor(),
            unlabeled_ratio=self.config.getfloat("Training", "unlabeled_ratio")
        )

    def load_test_data(self):
        from semi_supervised_dataset import SemiSupervisedCUB200
        return SemiSupervisedCUB200(
            root='CUB-200',
            train=False,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(self.config.getint("Training", "image_size"), scale=(0.8, 1.0), ratio=(0.75, 1.333)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
            ]),
            unlabeled_ratio=0.0
        )

    def create_contrastive_dataloader(self, dataset, aug1, aug2, batch_size, shuffle, num_workers, pin_memory, prefetch_factor):
        from contrastive_dataset import ContrastiveDataset
        contrastive_dataset = ContrastiveDataset(dataset, aug1, aug2)
        return torch.utils.data.DataLoader(
            contrastive_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )

    def create_dataloaders(self):
        train_dataset = self.load_train_data()
        test_dataset = self.load_test_data()
        aug1_transform = self.get_augmentation_transforms()
        aug2_transform = self.get_augmentation_transforms()
        inverse_aug1_transform = self.get_inverse_transforms()
        inverse_aug2_transform = self.get_inverse_transforms()

        train_loader = self.create_contrastive_dataloader(
            dataset=train_dataset,
            aug1=aug1_transform,
            aug2=aug2_transform,
            batch_size=self.config.getint("Training", "batch_size"),
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.getint("Training", "batch_size"),
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2
        )
        return train_loader, test_loader, inverse_aug1_transform, inverse_aug2_transform

# --- 3. 模型定义模块 ---
class ModelInitializer:
    def __init__(self, config):
        self.config = config

    def initialize_student(self):
        student_net = models.resnet18(pretrained=True)
        student_net.fc = nn.Linear(student_net.fc.in_features, self.config.getint("Training", "num_classes"))
        return student_net

    def initialize_teacher(self):
        teacher_net = models.resnet50(pretrained=True)
        num_ftrs = teacher_net.fc.in_features
        teacher_net.fc = nn.Linear(num_ftrs, self.config.getint("Training", "num_classes"))
        for param in teacher_net.parameters():
            param.requires_grad = False
        return teacher_net

    def initialize_compression_layer(self):
        compression_layer = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self._init_weights(compression_layer)
        return compression_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# --- 4. 损失函数和评估模块 ---
def safe_consistency_loss(pred, target, epsilon=1e-8):
    if torch.isnan(pred).any() or torch.isnan(target).any():
        print("Warning: NaN detected in consistency loss inputs")
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return F.smooth_l1_loss(pred / (pred.norm(dim=1, keepdim=True) + epsilon),
                            target / (target.norm(dim=1, keepdim=True) + epsilon))

def evaluate(model, test_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.mean().item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = loss_total / len(test_loader)
    return accuracy, avg_loss

def get_features(model, x, layer_name):
    if layer_name == 'layer4':
        return model.layer4(x)
    elif layer_name == 'layer3':
        return model.layer3(x)
    else:
        raise ValueError(f"Invalid layer name: {layer_name}")

# --- 5. 训练逻辑模块 ---
def train_model(config, student_net, teacher_net, compression_layer, train_loader, test_loader, device, inverse_aug1_transform, inverse_aug2_transform):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(
        list(student_net.parameters()) + list(compression_layer.parameters()),
        lr=config.getfloat("Training", "lr")
    )
    def warmup_lambda(epoch):
        if epoch < config.getint("Training", "warmup_epochs"):
            return epoch / config.getint("Training", "warmup_epochs")
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    model_save_path = 'studentmodel_best.pth'
    global_step = 0

    # NaN detection hook
    def nan_detector_hook(module, input, output):
        if torch.isnan(output).any():
            print(f"NaN detected in the output of {module}")

    student_net.layer4.register_forward_hook(nan_detector_hook)
    teacher_net.layer4.register_forward_hook(nan_detector_hook)

    for epoch in range(config.getint("Training", "epochs")):
        student_net.train()
        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for batch_idx, contrastive_batch in enumerate(train_loader):
            original_images = contrastive_batch['original'].to(device)
            aug1_images = contrastive_batch['aug1'].to(device)
            aug2_images = contrastive_batch['aug2'].to(device)
            labels = contrastive_batch['label'].to(device)

            unlabeled_mask = (labels == -1)
            w = torch.ones(labels.size(0), device=device)

            if unlabeled_mask.any():
                with torch.no_grad():
                    logits_t1 = teacher_net(aug1_images[unlabeled_mask])
                    logits_t2 = teacher_net(aug2_images[unlabeled_mask])
                    p1 = F.softmax(logits_t1, dim=1)
                    p2 = F.softmax(logits_t2, dim=1)
                    diff = (p1 - p2).pow(2).sum(dim=1).sqrt()
                    w_unlabeled = torch.exp(-config.getfloat("Training", "alpha") * diff)
                    p_avg = 0.5 * (p1 + p2)
                    pseudo_labels = p_avg.argmax(dim=1)

                labels[unlabeled_mask] = pseudo_labels
                w[unlabeled_mask] = w_unlabeled

            optimizer.zero_grad()
            student_logits = student_net(aug1_images)
            ce_all = criterion(student_logits, labels)
            loss_cls = (ce_all * w).mean()

            _, predicted = torch.max(student_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            Fs = get_features(student_net, aug1_images, config.get("Training", "layer_name"))
            Ft = get_features(teacher_net, aug2_images, config.get("Training", "layer_name"))
            Ft_compressed = compression_layer(Ft)
            invaug1_Fs = inverse_aug1_transform(Fs)
            invaug2_Ft = inverse_aug2_transform(Ft_compressed)
            loss_cons = safe_consistency_loss(invaug2_Ft, invaug1_Fs)

            loss_total = loss_cls + 0.1 * loss_cons
            train_loss_total += loss_total.item()

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), config.getfloat("Training", "gradient_clip_norm"))
            torch.nn.utils.clip_grad_norm_(compression_layer.parameters(), config.getfloat("Training", "gradient_clip_norm"))
            optimizer.step()

            if batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_cls_loss": loss_cls.item(),
                    "train/batch_cons_loss": loss_cons.item(),
                    "train/batch_w_mean": w.mean().item(),
                    "train/batch_total_loss": loss_total.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                print(f"Epoch [{epoch + 1}/{config.getint('Training', 'epochs')}], Batch [{batch_idx}], "
                      f"ClsLoss: {loss_cls.item():.4f}, ConsLoss: {loss_cons.item():.4f}, "
                      f"w_mean: {w.mean().item():.4f}, TotalLoss: {loss_total.item():.4f}")
            global_step += 1

        accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss_total / len(train_loader)
        scheduler.step()

        wandb.log({
            "val/accuracy": accuracy,
            "val/loss": avg_loss,
            "train/accuracy": train_accuracy,
            "train/loss": avg_train_loss,
            "train/epoch": epoch
        }, step=global_step)
        print(f"Epoch [{epoch + 1}/{config.getint('Training', 'epochs')}], "
              f"Train Accuracy: {train_accuracy:.2f}%, Train Loss: {avg_train_loss:.4f}, "
              f"Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

        improvement = accuracy - best_val_accuracy
        if improvement >= config.getfloat("Training", "improvement_threshold"):
            best_val_accuracy = accuracy
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }, model_save_path)
            wandb.log({"best_val_accuracy": best_val_accuracy}, step=global_step)
            print(f"Validation accuracy improved, saving model to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config.getint("Training", "patience"):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training finished!")
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best student model weights from '{model_save_path}'.")
        wandb.save(model_save_path)

    accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
    print(f"Final Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
    wandb.log({"final_test_accuracy": accuracy, "final_test_loss": avg_loss})

# --- 6. 日志记录模块 ---
def log_gradients(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            wandb.log({f"gradients/{name}_norm": grad_norm}, step=step)

def log_feature_stats(tensor, name, step):
    wandb.log({
        f"features/{name}_mean": tensor.mean().item(),
        f"features/{name}_std": tensor.std().item(),
        f"features/{name}_max": tensor.max().item(),
        f"features/{name}_min": tensor.min().item()
    }, step=step)

# --- 主程序 ---
def main():
    torch.multiprocessing.freeze_support()
    args = parse_args()
    config = load_config(args.config)
    wandb.init(project="semi-supervised-learning", config=config)
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化各个模块
    dataset_loader = DatasetLoader(config)
    train_loader, test_loader, inverse_aug1_transform, inverse_aug2_transform = dataset_loader.create_dataloaders()

    model_initializer = ModelInitializer(config)
    student_net = model_initializer.initialize_student().to(device)
    teacher_net = model_initializer.initialize_teacher().to(device)
    compression_layer = model_initializer.initialize_compression_layer().to(device)

    # 加载 TeacherNet 权重
    teacher_weights_path = 'model_checkpoints/best_model.pth'
    if os.path.exists(teacher_weights_path):
        checkpoint = torch.load(teacher_weights_path)
        try:
            teacher_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded TeacherNet weights from '{teacher_weights_path}'.")
        except RuntimeError as e:
            print(f"Error loading TeacherNet weights: {e}")
            return
    else:
        print("No custom TeacherNet checkpoint found, using initialized pretrained weights.")

    train_model(config, student_net, teacher_net, compression_layer, train_loader, test_loader, device, inverse_aug1_transform, inverse_aug2_transform)

    wandb.finish()

if __name__ == "__main__":
    main()