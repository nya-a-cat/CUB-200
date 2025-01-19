import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.v2 as transforms

from semi_supervised_dataset import SemiSupervisedCUB200
from contrastive_dataset import create_contrastive_dataloader
from custom_transforms import get_augmentation_transforms, get_inverse_transforms
from utils import consistency_loss, get_features

def main():
    torch.multiprocessing.freeze_support()

    # --- Hyperparameters and Configurations ---
    batch_size = 200
    image_size = 224
    num_classes = 200
    layer_name = 'layer4'
    unlabeled_ratio = 0.6  # 60%的无标签数据

    # 训练轮数和学习率，仅作示例
    epochs = 2
    lr = 1e-3

    # --- Data Loaders ---
    aug1_transform, aug2_transform = get_augmentation_transforms(size=image_size)
    inverse_aug1_transform, inverse_aug2_transform = get_inverse_transforms()

    # 使用半监督版本的 CUB200 数据集：部分数据label=-1
    train_dataset = SemiSupervisedCUB200(
        root='CUB-200',
        train=True,
        transform=transforms.ToDtype,  # 这里保留你的自定义 transform
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
        unlabeled_ratio=0.0  # 测试集保持全标签
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
    # 进入简单的训练循环
    student_net.train()
    for epoch in range(epochs):
        for batch_idx, contrastive_batch in enumerate(train_dataloader):

            original_images = contrastive_batch['original'].to(device)
            aug1_images = contrastive_batch['aug1'].to(device)
            aug2_images = contrastive_batch['aug2'].to(device)
            labels = contrastive_batch['label'].to(device)  # 其中可能包含 -1 表示无标签

            # ----------------------------------------------------
            # 1) 对无标签的样本生成伪标签
            # ----------------------------------------------------
            unlabeled_mask = (labels == -1)
            if unlabeled_mask.any():
                with torch.no_grad():
                    # 用 teacher_net 对无标签的 aug2_images 做推理(也可用 original_images 或 aug1_images)
                    teacher_logits = teacher_net(aug2_images[unlabeled_mask])
                    pseudo_labels = teacher_logits.argmax(dim=1)
                # 将 -1 替换为 teacher 推理得到的类别
                labels[unlabeled_mask] = pseudo_labels

            # ----------------------------------------------------
            # 2) 计算分类损失 (Student 对 aug1_images)
            # ----------------------------------------------------
            student_logits = student_net(aug1_images)
            loss_cls = F.cross_entropy(student_logits, labels)

            # ----------------------------------------------------
            # 3) 计算一致性损失 (在特征层做对齐)
            # ----------------------------------------------------
            Fs = get_features(student_net, aug1_images, layer_name)
            Ft = get_features(teacher_net, aug2_images, layer_name)
            Ft_compressed = compression_layer(Ft)

            invaug1_Fs = inverse_aug1_transform(Fs)
            invaug2_Ft = inverse_aug2_transform(Ft_compressed)

            loss_cons = consistency_loss(invaug2_Ft, invaug1_Fs)

            # 可以给一致性损失一个权重，比如 0.1、0.01等
            loss_total = loss_cls + 0.1 * loss_cons

            # ----------------------------------------------------
            # 4) 反向传播 & 更新
            # ----------------------------------------------------
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], "
                    f"Cls Loss: {loss_cls.item():.4f}, Cons Loss: {loss_cons.item():.4f}, "
                    f"Total Loss: {loss_total.item():.4f}"
                )

    print("Training finished!")

    # 后面如果要做测试 / 验证，可继续对 test_loader 做评估
    # ...

if __name__ == "__main__":
    main()
