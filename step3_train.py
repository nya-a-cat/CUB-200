import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import os
import wandb

from semi_supervised_dataset import SemiSupervisedCUB200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import get_features, visualize_pseudo_labels


def init_weights(m):
    """Initialize network weights using Kaiming initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def log_gradients(model, step):
    """Log gradient norms to wandb"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            wandb.log({f"gradients/{name}_norm": grad_norm}, step=step)


def log_feature_stats(tensor, name, step):
    """Log feature statistics to wandb"""
    wandb.log({
        f"features/{name}_mean": tensor.mean().item(),
        f"features/{name}_std": tensor.std().item(),
        f"features/{name}_max": tensor.max().item(),
        f"features/{name}_min": tensor.min().item()
    }, step=step)


def safe_consistency_loss(pred, target, epsilon=1e-8):
    """Compute consistency loss with safety checks"""
    if torch.isnan(pred).any() or torch.isnan(target).any():
        print("Warning: NaN detected in consistency loss inputs")
        return torch.tensor(0.0, requires_grad=True, device=pred.device)

    # Use smooth L1 loss instead of MSE for better numerical stability
    return F.smooth_l1_loss(pred / (pred.norm(dim=1, keepdim=True) + epsilon),
                            target / (target.norm(dim=1, keepdim=True) + epsilon))


def evaluate(model, test_loader, device, criterion):
    """Evaluate model performance"""
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


def main():
    torch.multiprocessing.freeze_support()

    # --- Hyperparameters and Configurations ---
    config = {
        "batch_size": 150,
        "image_size": 224,
        "num_classes": 200,
        "layer_name": 'layer4',
        "unlabeled_ratio": 0.6,
        "epochs": 100,
        "lr": 5e-4,  # Reduced learning rate for stability
        "patience": 10,
        "improvement_threshold": 1.0,
        "alpha": 5.0,
        "warmup_epochs": 5,  # Added warmup epochs
        "gradient_clip_norm": 1.0,  # Added gradient clipping
    }

    # --- Initialize WandB ---
    wandb.init(project="semi-supervised-learning", config=config)
    config = wandb.config

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=config.image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    train_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=True,
        transform=transforms.ToTensor(),
        unlabeled_ratio=config.unlabeled_ratio
    )
    train_dataloader = create_contrastive_dataloader(
        dataset=train_dataset,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    test_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=False,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ]),
        unlabeled_ratio=0.0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # --- Model Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Student & Teacher
    student_net = models.resnet18(pretrained=True)
    teacher_net = models.resnet50(pretrained=True)

    # Modify final layers
    num_ftrs = teacher_net.fc.in_features
    teacher_net.fc = nn.Linear(num_ftrs, config.num_classes)
    student_net.fc = nn.Linear(student_net.fc.in_features, config.num_classes)

    # Freeze teacher parameters
    for param in teacher_net.parameters():
        param.requires_grad = False

    # Initialize teacher network
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

    # Optimized compression layer with normalization and activation
    compression_layer = nn.Sequential(
        nn.BatchNorm2d(2048),
        nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )
    compression_layer.apply(init_weights)

    # Move models to device
    student_net.to(device)
    teacher_net.to(device)
    compression_layer.to(device)

    # Optimizer with warmup
    optimizer = torch.optim.Adam(
        list(student_net.parameters()) + list(compression_layer.parameters()),
        lr=config.lr
    )

    # Learning rate scheduler with warmup
    def warmup_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    teacher_net.eval()
    student_net.train()

    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_state = None
    model_save_path = 'studentmodel_best.pth'
    global_step = 0

    for epoch in range(config.epochs):
        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for batch_idx, contrastive_batch in enumerate(train_dataloader):
            original_images = contrastive_batch['original'].to(device)
            aug1_images = contrastive_batch['aug1'].to(device)
            aug2_images = contrastive_batch['aug2'].to(device)
            labels = contrastive_batch['label'].to(device)

            # Handle unlabeled data
            unlabeled_mask = (labels == -1)
            w = torch.ones(labels.size(0), device=device)

            if unlabeled_mask.any():
                with torch.no_grad():
                    aug1_unlabeled = aug1_images[unlabeled_mask]
                    aug2_unlabeled = aug2_images[unlabeled_mask]
                    logits_t1 = teacher_net(aug1_unlabeled)
                    logits_t2 = teacher_net(aug2_unlabeled)
                    p1 = F.softmax(logits_t1, dim=1)
                    p2 = F.softmax(logits_t2, dim=1)
                    diff = (p1 - p2).pow(2).sum(dim=1).sqrt()
                    w_unlabeled = torch.exp(-config.alpha * diff)
                    p_avg = 0.5 * (p1 + p2)
                    pseudo_labels = p_avg.argmax(dim=1)

                labels[unlabeled_mask] = pseudo_labels
                w[unlabeled_mask] = w_unlabeled

            # Forward pass through student network
            student_logits = student_net(aug1_images)
            ce_all = criterion(student_logits, labels)
            loss_cls = (ce_all * w).mean()

            # Track accuracy
            _, predicted = torch.max(student_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Get features and apply consistency loss
            Fs = get_features(student_net, aug1_images, config.layer_name)
            Ft = get_features(teacher_net, aug2_images, config.layer_name)

            # Log feature statistics
            log_feature_stats(Fs, "student_features", global_step)
            log_feature_stats(Ft, "teacher_features", global_step)

            # Apply compression and compute consistency loss
            Ft_compressed = compression_layer(Ft)
            invaug1_Fs = inverse_aug1_transform(Fs)
            invaug2_Ft = inverse_aug2_transform(Ft_compressed)
            loss_cons = safe_consistency_loss(invaug2_Ft, invaug1_Fs)

            # Total loss and backward pass
            loss_total = loss_cls + 0.1 * loss_cons
            train_loss_total += loss_total.item()

            optimizer.zero_grad()
            loss_total.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), config.gradient_clip_norm)
            torch.nn.utils.clip_grad_norm_(compression_layer.parameters(), config.gradient_clip_norm)

            optimizer.step()

            # Log gradients
            log_gradients(student_net, global_step)
            log_gradients(compression_layer, global_step)

            # Logging
            if batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_cls_loss": loss_cls.item(),
                    "train/batch_cons_loss": loss_cons.item(),
                    "train/batch_w_mean": w.mean().item(),
                    "train/batch_total_loss": loss_total.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}], Batch [{batch_idx}], "
                    f"ClsLoss: {loss_cls.item():.4f}, ConsLoss: {loss_cons.item():.4f}, "
                    f"w_mean: {w.mean().item():.4f}, TotalLoss: {loss_total.item():.4f}"
                )

            global_step += 1

        # Evaluate and update learning rate
        accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss_total / len(train_dataloader)

        scheduler.step()

        # Logging
        wandb.log({
            "val/accuracy": accuracy,
            "val/loss": avg_loss,
            "train/accuracy": train_accuracy,
            "train/loss": avg_train_loss,
            "train/epoch": epoch
        }, step=global_step)

        print(f"Epoch [{epoch + 1}/{config.epochs}], "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Test Accuracy: {accuracy:.2f}%, "
              f"Test Loss: {avg_loss:.4f}")

        # Model checkpointing
        improvement = accuracy - best_val_accuracy
        if improvement >= config.improvement_threshold:
            best_val_accuracy = accuracy
            epochs_no_improve = 0
            best_model_state = student_net.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }, model_save_path)
            wandb.log({"best_val_accuracy": best_val_accuracy}, step=global_step)
            print(f"Validation accuracy improved, saving model to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training finished!")

    # Load best model and evaluate
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best student model weights from '{model_save_path}'.")
        wandb.save(model_save_path)

    # Final evaluation
    accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
    print(f"Final Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
    wandb.log({"final_test_accuracy": accuracy, "final_test_loss": avg_loss})

    # Cleanup
    wandb.finish()


if __name__ == "__main__":
    main()