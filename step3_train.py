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
from utils import consistency_loss, get_features, visualize_pseudo_labels

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
        "lr": 1e-3,
        "patience": 10,
        "improvement_threshold": 1.0,
        "alpha": 5.0
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
        transform=transforms.Compose([transforms.transforms.ToImage(), transforms.transforms.ToDtype(torch.float32, scale=True)]),
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
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),  # 随机裁剪并缩放到目标尺寸
            transforms.RandomRotation(degrees=15),  # 随机旋转，角度范围 +/- 15 度
            transforms.ToImage(),
            transforms.transforms.ToDtype(torch.float32, scale=True),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
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

    # --- 模型初始化 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Student & Teacher
    student_net = models.resnet18(weights='IMAGENET1K_V1')
    teacher_net = models.resnet50(weights='IMAGENET1K_V1')

    # Modify the final fully connected layer of the teacher network before loading weights
    num_ftrs = teacher_net.fc.in_features
    teacher_net.fc = nn.Linear(num_ftrs, config.num_classes)

    for param in teacher_net.parameters():
        param.requires_grad = False

    # --- Initialize TeacherNet ---
    teacher_weights_path = 'model_checkpoints/best_model.pth'
    if os.path.exists(teacher_weights_path):
        checkpoint = torch.load(teacher_weights_path)

        # --- Load TeacherNet weights with error handling ---
        try:
            teacher_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded TeacherNet weights from '{teacher_weights_path}'.")
        except RuntimeError as e:
            print(f"Error loading TeacherNet weights: {e}")
            print("Please ensure the checkpoint was saved with a compatible ResNet50 architecture and the correct number of output classes.")
            return  # Exit if there's an error
    else:
        print("No custom TeacherNet checkpoint found, using initialized pretrained weights from torchvision.")
    teacher_net.to(device).eval()

    # 替换最后一层，全连接输出 200 类
    student_net.fc = nn.Linear(student_net.fc.in_features, config.num_classes)

    # # 1x1 卷积层
    # compression_layer = nn.Contransformsd(in_channels=2048, out_channels=512, kernel_size=1)

    # 1. 在compression layer前后添加归一化层
    compression_layer = nn.Sequential(
        nn.BatchNorm2d(2048),  # 输入归一化
        nn.Contransformsd(in_channels=2048, out_channels=512, kernel_size=1),
        nn.BatchNorm2d(512)  # 输出归一化
    )

    # 2. 添加梯度裁剪
    # torch.nn.utils.clip_grad_norm_(compression_layer.parameters(), max_norm=1.0)

    student_net.to(device)
    teacher_net.to(device)
    compression_layer.to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        list(student_net.parameters()) + list(compression_layer.parameters()),
        lr=config.lr
    )

    teacher_net.eval()
    student_net.train()

    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_state = None
    model_save_path = 'studentmodel_best.pth'

    # Flag to temporarily disable compression layer for debugging
    disable_compression_layer = False

    for epoch in range(config.epochs):
        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for batch_idx, contrastive_batch in enumerate(train_dataloader):
            original_images = contrastive_batch['original'].to(device)
            aug1_images = contrastive_batch['aug1'].to(device)
            aug2_images = contrastive_batch['aug2'].to(device)
            labels = contrastive_batch['label'].to(device)

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

            student_logits = student_net(aug1_images)
            ce_all = criterion(student_logits, labels)
            loss_cls = (ce_all * w).mean()

            _, predicted = torch.max(student_logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            Fs = get_features(student_net, aug1_images, config.layer_name)
            Ft = get_features(teacher_net, aug2_images, config.layer_name)

            # --- Debugging Compression Layer ---
            print(f"Epoch [{epoch+1}/{config.epochs}], Batch [{batch_idx}] - Before Compression Layer")
            # print("Compression Layer Weight:", compression_layer.weight)
            # print("Compression Layer Bias:", compression_layer.bias)
            # print("Compression Layer Weight Grad:", compression_layer.weight.grad)
            # print("Compression Layer Bias Grad:", compression_layer.bias.grad)
            print("Min of Ft:", torch.min(Ft))
            print("Max of Ft:", torch.max(Ft))
            print("Is NaN in Ft:", torch.isnan(Ft).any())

            if not disable_compression_layer:
                Ft_compressed = compression_layer(Ft)
                print("Is NaN in Ft_compressed:", torch.isnan(Ft_compressed).any())
            else:
                Ft_compressed = Ft
                print("Compression layer disabled for debugging.")

            invaug1_Fs = inverse_aug1_transform(Fs)
            invaug2_Ft = inverse_aug2_transform(Ft_compressed)
            loss_cons = consistency_loss(invaug2_Ft, invaug1_Fs)
            # 输出训练的第几批次和 loss_cons
            print(f"Epoch [{epoch + 1}/{config.epochs}], Batch [{batch_idx}], Consistency Loss: {loss_cons.item():.4f}")

            loss_total = loss_cls + 0.1 * loss_cons

            # --- 添加 NaN 检测 ---
            if torch.isnan(loss_total):
                print("Error: NaN detected in loss. Stopping training.")
                break  # 停止当前 batch 的训练
            # --- NaN 检测结束 ---

            train_loss_total += loss_total.item()

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # --- Gradient Clipping (Optional) ---
            # torch.nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=1)
            # torch.nn.utils.clip_grad_norm_(compression_layer.parameters(), max_norm=1)

            if batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_cls_loss": loss_cls.item(),
                    "train/batch_cons_loss": loss_cons.item(),
                    "train/batch_w_mean": w.mean().item(),
                    "train/batch_total_loss": loss_total.item()
                })
                print(
                    f"Epoch [{epoch+1}/{config.epochs}], Batch [{batch_idx}], "
                    f"ClsLoss: {loss_cls.item():.4f}, ConsLoss: {loss_cons.item():.4f}, "
                    f"w_mean: {w.mean().item():.4f}, TotalLoss: {loss_total.item():.4f}"
                )

        # 在 epoch 结束时检查是否因为 NaN 而停止
        if torch.isnan(loss_total):
            break  # 停止整个 epoch 的训练

        accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss_total / len(train_dataloader)

        wandb.log({
            "val/accuracy": accuracy,
            "val/loss": avg_loss,
            "train/accuracy": train_accuracy,
            "train/loss": avg_train_loss
        })
        print(f"Epoch [{epoch+1}/{config.epochs}], Train Accuracy: {train_accuracy:.2f}%, Train Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

        improvement = accuracy - best_val_accuracy
        if improvement >= config.improvement_threshold:
            best_val_accuracy = accuracy
            epochs_no_improve = 0
            best_model_state = student_net.state_dict()
            torch.save({'model_state_dict': best_model_state}, model_save_path)
            wandb.log({"best_val_accuracy": best_val_accuracy})
            print(f"Validation accuracy improved, saving model to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("Training finished!")

    # Load the best student model
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        student_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best student model weights from '{model_save_path}'.")
        wandb.save(model_save_path)

    # Visualize pseudo-labels
    visualize_pseudo_labels(
        teacher_net=teacher_net,
        dataset=train_dataset,
        device=device,
        layer_name=config.layer_name,
        sample_size=500,
        alpha=config.alpha
    )

    # Evaluate the best student model
    accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
    print(f"Test Accuracy with best model: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
    wandb.log({"final_test_accuracy": accuracy, "final_test_loss": avg_loss})

    wandb.finish()

if __name__ == "__main__":
    main()