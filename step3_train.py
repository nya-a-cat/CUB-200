import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as transforms

from semi_supervised_dataset import SemiSupervisedCUB200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import consistency_loss, get_features, visualize_pseudo_labels

def evaluate(model, test_loader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    loss_total = 0

    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss_total += loss.item()

    accuracy = 100 * correct / total
    avg_loss = loss_total / len(test_loader)

    return accuracy, avg_loss

def main():
    torch.multiprocessing.freeze_support()

    # --- Hyperparameters and Configurations ---
    batch_size = 200
    image_size = 224
    num_classes = 200
    layer_name = 'layer4'
    unlabeled_ratio = 0.6  # 60%的无标签数据

    epochs = 2
    lr = 1e-3

    # 置信度相关超参数
    alpha = 5.0  # 控制差异->置信度的衰减

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    # 使用半监督版本的 CUB200 数据集：部分数据 label=-1
    train_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=True,
        transform=transforms.ToDtype,  # 你的自定义 transform
        unlabeled_ratio=unlabeled_ratio
    )
    train_dataloader = create_contrastive_dataloader(
        dataset=train_dataset,
        aug1=aug1_transform,
        aug2=aug2_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # 测试集：保持全标签，和原来一样
    test_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=False,
        transform=transforms.ToDtype,
        unlabeled_ratio=0.0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )

    # --- 模型初始化 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Student & Teacher
    student_net = models.resnet18(pretrained=True)
    teacher_net = models.resnet18(pretrained=True)
    # 冻结teacher网络参数
    for param in teacher_net.parameters():
        param.requires_grad = False

    # 替换最后一层，全连接输出 200 类
    student_net.fc = nn.Linear(512, num_classes)
    teacher_net.fc = nn.Linear(512, num_classes)

    # 1x1 卷积层，用于将 teacher 的特征通道压缩到 student 的特征通道数（如果需要）
    teacher_feature_dim = teacher_net._modules[layer_name][-1].conv2.out_channels
    student_feature_dim = student_net._modules[layer_name][-1].conv2.out_channels
    compression_layer = nn.Conv2d(teacher_feature_dim, student_feature_dim, kernel_size=1)

    student_net.to(device)
    teacher_net.to(device)
    compression_layer.to(device)

    # 优化器仅更新 student_net（和 compression_layer，如果需要）参数
    optimizer = torch.optim.Adam(
        list(student_net.parameters()) + list(compression_layer.parameters()),
        lr=lr
    )

    # Teacher 只做推理，不需要 train
    teacher_net.eval()
    student_net.train()

    # 使用 'none' 形式的交叉熵以便对每个样本加权
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(epochs):
        for batch_idx, contrastive_batch in enumerate(train_dataloader):

            original_images = contrastive_batch['original'].to(device)
            aug1_images = contrastive_batch['aug1'].to(device)
            aug2_images = contrastive_batch['aug2'].to(device)
            labels = contrastive_batch['label'].to(device)  # 可能包含 -1 表示无标签

            # ----------------------------------------------------------------
            # (1) 对无标签的样本，用 TeacherNet 在 aug1 & aug2 上做推理 -> 算差异 -> 得置信度 w -> 生成伪标签
            # ----------------------------------------------------------------
            unlabeled_mask = (labels == -1)
            w = torch.ones(labels.size(0), device=device)  # 初始时，对整批样本的权重都设为1

            if unlabeled_mask.any():
                with torch.no_grad():
                    # 先拿到无标签样本的 aug1 & aug2
                    aug1_unlabeled = aug1_images[unlabeled_mask]
                    aug2_unlabeled = aug2_images[unlabeled_mask]

                    # TeacherNet 两次推理
                    logits_t1 = teacher_net(aug1_unlabeled)
                    logits_t2 = teacher_net(aug2_unlabeled)
                    p1 = F.softmax(logits_t1, dim=1)
                    p2 = F.softmax(logits_t2, dim=1)

                    # 计算两次预测差异
                    diff = (p1 - p2).pow(2).sum(dim=1).sqrt()  # shape=[#unlabeled]
                    # 映射到置信度 w_unlabeled
                    w_unlabeled = torch.exp(-alpha * diff)  # (0,1]

                    # 生成伪标签 (对p1,p2做平均再argmax)
                    p_avg = 0.5 * (p1 + p2)
                    pseudo_labels = p_avg.argmax(dim=1)

                # 将无标签对应位置的标签替换为伪标签
                labels[unlabeled_mask] = pseudo_labels
                # 将无标签对应位置的权重替换为 w_unlabeled
                w[unlabeled_mask] = w_unlabeled

            # ----------------------------------------------------------------
            # (2) 计算分类损失: Student 对 aug1_images -> 交叉熵 -> 加权
            # ----------------------------------------------------------------
            student_logits = student_net(aug1_images)
            ce_all = criterion(student_logits, labels)  # shape=[batch_size]
            # 逐样本乘以权重 w，再求均值
            loss_cls = (ce_all * w).mean()

            # ----------------------------------------------------------------
            # (3) 一致性损失: (在特征层对齐 Teacher vs Student)
            # ----------------------------------------------------------------
            Fs = get_features(student_net, aug1_images, layer_name)
            Ft = get_features(teacher_net, aug2_images, layer_name)
            Ft_compressed = compression_layer(Ft)

            invaug1_Fs = inverse_aug1_transform(Fs)
            invaug2_Ft = inverse_aug2_transform(Ft_compressed)

            loss_cons = consistency_loss(invaug2_Ft, invaug1_Fs)

            # 可给一致性损失一个权重，比如 0.1
            loss_total = loss_cls + 0.1 * loss_cons

            # ----------------------------------------------------------------
            # (4) 反向传播 & 更新
            # ----------------------------------------------------------------
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], "
                    f"ClsLoss: {loss_cls.item():.4f}, ConsLoss: {loss_cons.item():.4f}, "
                    f"w_mean: {w.mean().item():.4f}, TotalLoss: {loss_total.item():.4f}"
                )

    print("Training finished!")

    # ====== 训练结束，做可视化 ======
    # 注意，这里我们把 train_dataset 传给可视化函数，因为它包含了无标签数据
    # 并指定我们要可视化 teacher_net 的伪标签
    visualize_pseudo_labels(
        teacher_net=teacher_net,
        dataset=train_dataset,
        device=device,
        layer_name=layer_name,  # 例如 'layer4'
        sample_size=500,  # 可以调整采样数量
        alpha=alpha  # 和训练时一致
    )

    # 后续再做 test_loader 评估等 ...
    accuracy, avg_loss = evaluate(student_net, test_loader, device, criterion)
    print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
