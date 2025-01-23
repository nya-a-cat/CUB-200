import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import os
import wandb
from torchvision.utils import save_image
import tqdm
import argparse

from semi_supervised_dataset import SemiSupervisedCUB200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import consistency_loss, get_features, visualize_pseudo_labels


# --- Configuration ---
def get_default_config(lr=1e-3, unlabeled_ratio=0.6):
    config = {
        "batch_size": 200,
        "image_size": 224,
        "num_classes": 200,
        "layer_name": 'layer4',
        "unlabeled_ratio": unlabeled_ratio,
        "epochs": 100,
        "lr": lr,
        "patience": 100,
        "improvement_threshold": 1.0,
        "alpha": 5.0,
        "project_name": "semi-supervised-learning",
        "teacher_weights_path": 'model_checkpoints/best_model.pth',
        "student_model_save_path": 'studentmodel_best.pth',
        "feature_maps_dir": "feature_maps",
        "cons_loss_factor": 1
    }
    return config


# --- Data Loading ---
def create_data_loaders(config):
    aug1_transform, aug2_transform = get_augmentation_transforms(size=config["image_size"])
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    train_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=True,
        transform=transforms.ToTensor(),
        unlabeled_ratio=config["unlabeled_ratio"]
    )
    train_dataloader = create_contrastive_dataloader(
        dataset=train_dataset,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    test_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]),
        unlabeled_ratio=0.0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    return train_dataloader, test_loader, inverse_aug1_transform, inverse_aug2_transform


# --- Model Initialization ---
def initialize_models(config, device):
    student_net = models.resnet18(pretrained=True).float()
    teacher_net = models.resnet50(pretrained=True).float()

    num_ftrs_teacher = teacher_net.fc.in_features
    teacher_net.fc = nn.Linear(num_ftrs_teacher, config["num_classes"]).float()
    for param in teacher_net.parameters():
        param.requires_grad = False

    if os.path.exists(config["teacher_weights_path"]):
        checkpoint = torch.load(config["teacher_weights_path"])
        try:
            teacher_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded TeacherNet weights from '{config['teacher_weights_path']}'.")
        except RuntimeError as e:
            print(f"Error loading TeacherNet weights: {e}")
            print("Ensure checkpoint is compatible with ResNet50 and output classes.")
            return None, None, None
    else:
        print("No custom TeacherNet checkpoint found, using pretrained weights.")

    student_net.fc = nn.Linear(student_net.fc.in_features, config["num_classes"]).float()

    compression_layer = nn.Sequential(
        nn.BatchNorm2d(2048),
        nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
        nn.BatchNorm2d(512)
    ).float()

    student_net.to(device)
    teacher_net.to(device).eval()
    compression_layer.to(device)

    return student_net, teacher_net, compression_layer


# --- Optimizer ---
def create_optimizer(student_net, compression_layer, config):
    optimizer = torch.optim.Adam(
        list(student_net.parameters()) + list(compression_layer.parameters()),
        lr=config["lr"]
    )
    return optimizer


# --- Training Step ---
def train_step(student_net, teacher_net, compression_layer, optimizer, train_dataloader, criterion, config, device,
               inverse_aug1_transform, inverse_aug2_transform, epoch, wandb_log, disable_compression_layer=False):
    student_net.train()
    train_loss_total = 0
    train_correct = 0
    train_total = 0

    progress_bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                             desc=f"Epoch [{epoch + 1}/{config['epochs']}]")

    for batch_idx, contrastive_batch in progress_bar:
        original_images = contrastive_batch['original'].to(device).float()
        aug1_images = contrastive_batch['aug1'].to(device).float()
        aug2_images = contrastive_batch['aug2'].to(device).float()
        labels = contrastive_batch['label'].to(device)

        try:
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
                    w_unlabeled = torch.exp(-config["alpha"] * diff)
                    p_avg = 0.5 * (p1 + p2)
                    pseudo_labels = p_avg.argmax(dim=1)

                labels[unlabeled_mask] = pseudo_labels
                w[unlabeled_mask] = w_unlabeled

            student_logits = student_net(aug1_images)
            ce_all = criterion(student_logits, labels)
            loss_cls = (ce_all * w).mean()

            _, predicted = torch.max(student_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            Fs = get_features(student_net, aug1_images, config["layer_name"])
            Ft = get_features(teacher_net, aug2_images, config["layer_name"])

            if not disable_compression_layer:
                Ft_compressed = compression_layer(Ft.float())
            else:
                Ft_compressed = Ft

            invaug1_Fs = inverse_aug1_transform(Fs)
            invaug2_Ft = inverse_aug2_transform(Ft_compressed)
            loss_cons = consistency_loss(invaug2_Ft, invaug1_Fs)

            loss_total = loss_cls + config["cons_loss_factor"] * loss_cons

            if torch.isnan(loss_total):
                print("Error: NaN detected in loss. Stopping training.")
                save_image(original_images, f"error_batch_{batch_idx}_original.png")
                save_image(aug1_images, f"error_batch_{batch_idx}_aug1.png")
                save_image(aug2_images, f"error_batch_{batch_idx}_aug2.png")
                return float('nan'), None, None, None, True, None

            train_loss_total += loss_total.item()

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                batch_log = {
                    "train/batch_cls_loss": loss_cls.item(),
                    "train/batch_cons_loss": loss_cons.item(),
                    "train/batch_w_mean": w.mean().item(),
                    "train/batch_total_loss": loss_total.item()
                }
                wandb_log_wrapper(batch_log)
                print(
                    f"Epoch [{epoch + 1}/{config['epochs']}], Batch [{batch_idx}], "
                    f"ClsLoss: {loss_cls.item():.4f}, ConsLoss: {loss_cons.item():.4f}, "
                    f"w_mean: {w.mean().item():.4f}, TotalLoss: {loss_total.item():.4f}"
                )
                progress_bar.set_postfix({
                    'ClsLoss': f'{loss_cls.item():.4f}',
                    'ConsLoss': f'{loss_cons.item():.4f}',
                    'TotalLoss': f'{loss_total.item():.4f}',
                    'Accuracy': f'{100 * train_correct / train_total:.2f}%' if train_total > 0 else 'N/A'
                })


        except Exception as e:
            print(f"Error during training at epoch {epoch}, batch {batch_idx}: {e}")
            save_image(original_images, f"error_batch_{batch_idx}_original.png")
            save_image(aug1_images, f"error_batch_{batch_idx}_aug1.png")
            save_image(aug2_images, f"error_batch_{batch_idx}_aug2.png")
            raise

    train_accuracy_epoch = 100 * train_correct / train_total
    avg_train_loss = train_loss_total / len(train_dataloader)
    return avg_train_loss, train_accuracy_epoch, student_net, optimizer, False, train_accuracy_epoch


# --- Evaluation Function (remains mostly the same) ---
def evaluate(model, test_loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device).float(), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.mean().item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = loss_total / len(test_loader)
    return accuracy, avg_loss


# --- Save Feature Map (remains the same) ---
def save_feature_map(feature_map, name, batch_idx, feature_maps_dir="feature_maps"):
    if not os.path.exists(feature_maps_dir):
        os.makedirs(feature_maps_dir)
    sample_feature_map = feature_map[0].cpu().detach()
    for i in range(min(8, sample_feature_map.shape[0])):
        channel_map = sample_feature_map[i]
        min_val = channel_map.min()
        max_val = channel_map.max()
        normalized_map = (channel_map - min_val) / (max_val - min_val + 1e-5)
        save_image(normalized_map, f"{feature_maps_dir}/batch_{batch_idx}_{name}_channel_{i}.png")


# --- WandB Logging Wrapper ---
def wandb_log_wrapper(log_dict):
    wandb.log(log_dict)


# --- Main Training Function ---
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader, test_loader, inverse_aug1_transform, inverse_aug2_transform = create_data_loaders(config)
    student_net, teacher_net, compression_layer = initialize_models(config, device)

    if student_net is None:
        print("Model initialization failed. Exiting.")
        return

    optimizer = create_optimizer(student_net, compression_layer, config)
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_state = None

    disable_compression_layer = False
    epoch_history = {}

    for epoch in range(config["epochs"]):
        avg_train_loss, train_accuracy, student_net, optimizer, nan_loss_detected, epoch_train_accuracy = \
            train_step(
                student_net, teacher_net, compression_layer, optimizer,
                train_dataloader, criterion, config, device,
                inverse_aug1_transform, inverse_aug2_transform, epoch,
                wandb_log_wrapper, disable_compression_layer
            )

        if nan_loss_detected:
            print("Training stopped due to NaN loss.")
            break

        if torch.isnan(torch.tensor(avg_train_loss)):
            print("Training stopped due to NaN loss after train step.")
            break

        accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)

        epoch_log = {
            "val/accuracy": accuracy,
            "val/loss": avg_loss,
            "train/accuracy": epoch_train_accuracy if epoch_train_accuracy is not None else train_accuracy,
            "train/loss": avg_train_loss
        }
        wandb_log_wrapper(epoch_log)
        train_accuracy_str = f"{epoch_train_accuracy:.2f}%" if epoch_train_accuracy is not None else 'N/A'

        print(
            f"Epoch [{epoch + 1}/{config['epochs']}], Train Accuracy: {train_accuracy_str}, Train Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

        improvement = accuracy - best_val_accuracy
        if improvement >= config["improvement_threshold"]:
            best_val_accuracy = accuracy
            epochs_no_improve = 0
            best_model_state = student_net.state_dict()
            torch.save({'model_state_dict': best_model_state}, config["student_model_save_path"])
            wandb_log_wrapper({"best_val_accuracy": best_val_accuracy})
            print(f"Validation accuracy improved, saving model to {config['student_model_save_path']}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config["patience"]:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        epoch_history[epoch + 1] = {
            'train_loss': avg_train_loss,
            'train_accuracy': epoch_train_accuracy if epoch_train_accuracy is not None else train_accuracy,
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    print("Training finished!")

    if os.path.exists(config["student_model_save_path"]):
        checkpoint = torch.load(config["student_model_save_path"])
        student_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best student model weights from '{config['student_model_save_path']}'.")
        wandb.save(config["student_model_save_path"])

    visualize_pseudo_labels(
        teacher_net=teacher_net,
        dataset=train_dataloader.dataset.dataset,
        device=device,
        layer_name=config["layer_name"],
        sample_size=500,
        alpha=config["alpha"]
    )

    accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
    print(f"Test Accuracy with best model: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
    wandb_log_wrapper({"final_test_accuracy": accuracy, "final_test_loss": avg_loss})

    wandb.log({"epoch_metrics": wandb.Table(columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'],
                                            data=[[epoch, metrics['train_loss'], metrics['train_accuracy'], metrics['val_loss'], metrics['val_accuracy']]
                                                  for epoch, metrics in epoch_history.items()])})


def main():
    torch.multiprocessing.freeze_support()
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description="Run experiments with different lr and unlabeled_ratio")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.6, help='Unlabeled ratio')
    args = parser.parse_args()

    config = get_default_config(lr=args.lr, unlabeled_ratio=args.unlabeled_ratio)

    wandb.init(project=config["project_name"], config=config,
               name=f"lr_{config['lr']}_ur_{config['unlabeled_ratio']}")
    config = wandb.config

    wandb.config.update({"cons_loss_factor": config["cons_loss_factor"]})

    train_model(config)

    wandb.finish()


if __name__ == "__main__":
    main()