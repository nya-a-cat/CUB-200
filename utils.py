# utils.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def tensor_to_image(tensor):
    # (C,H,W) -> (H,W,C)
    img = tensor.permute(1, 2, 0)
    img = img.cpu().numpy()
    # 归一化到0~1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def visualize_feature_map(feature_map):
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')

def apply_inverse_transform_to_tensor(tensor, inverse_transform):
    # 假设 inverse_transform 可以对 Tensor 直接操作
    return inverse_transform(tensor)

def visualize_consistency(original_image, aug1_image, aug2_image,
                          fs, ft_compressed, inv_aug1, inv_aug2, num_channels=4):
    invaug1_fs = apply_inverse_transform_to_tensor(fs, inv_aug1)
    invaug2_ft_compressed = apply_inverse_transform_to_tensor(ft_compressed, inv_aug2)

    fig, axes = plt.subplots(num_channels, 7, figsize=(12, 3 * num_channels))

    for i in range(num_channels):
        # Original
        ax = axes[i, 0]
        ax.imshow(tensor_to_image(original_image))
        if i == 0:
            ax.set_title('Original', fontsize=10)
        ax.axis('off')

        # Aug1
        ax = axes[i, 1]
        ax.imshow(tensor_to_image(aug1_image))
        if i == 0:
            ax.set_title('Aug1', fontsize=10)
        ax.axis('off')

        # Aug2
        ax = axes[i, 2]
        ax.imshow(tensor_to_image(aug2_image))
        if i == 0:
            ax.set_title('Aug2', fontsize=10)
        ax.axis('off')

        # Fs
        ax = axes[i, 3]
        ax.imshow(fs[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('Fs', fontsize=10)
        ax.axis('off')

        # invaug1(Fs)
        ax = axes[i, 4]
        ax.imshow(invaug1_fs[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('invaug1(Fs)', fontsize=10)
        ax.axis('off')

        # Ft_compressed
        ax = axes[i, 5]
        ax.imshow(ft_compressed[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('Ft_compressed', fontsize=10)
        ax.axis('off')

        # invaug2(Ft_compressed)
        ax = axes[i, 6]
        ax.imshow(invaug2_ft_compressed[0, i].detach().cpu().numpy(), cmap='viridis')
        if i == 0:
            ax.set_title('invaug2(Ft_compressed)', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def consistency_loss(invaug2_Ft, invaug1_Fs):
    return F.mse_loss(invaug2_Ft, invaug1_Fs)

def get_features(model, images, layer_name):
    """
    从指定的layer_name获取输出特征
    """
    features = {}
    def hook(module, input, output):
        features[layer_name] = output.detach()

    # 找到对应模块
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)

    handle = module.register_forward_hook(hook)
    _ = model(images)
    handle.remove()
    return features[layer_name]


def visualize_pseudo_labels(
        teacher_net,
        dataset,
        device,
        layer_name='layer4',
        sample_size=500,
        alpha=5.0
):
    """
    在给定的半监督数据集中，随机选取一定数量的无标签样本，用Teacher网路生成伪标签及置信度后可视化。
    :param teacher_net: 训练好的 Teacher 网络
    :param dataset: 半监督数据集 (SemiSupervisedCUB200)
    :param device: 'cuda' or 'cpu'
    :param layer_name: 指定提取特征的层名，比如 'layer4'
    :param sample_size: 随机采样多少张无标签图像进行可视化
    :param alpha: 和训练时一致，用于计算置信度 w = exp(- alpha * diff)
    """

    # 1. 从dataset中筛选无标签样本索引
    unlabeled_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == -1]
    if len(unlabeled_indices) == 0:
        print("Dataset中没有无标签样本，无法可视化伪标签。")
        return

    # 如果无标签数据过多，则做个随机抽样
    if len(unlabeled_indices) > sample_size:
        selected_indices = np.random.choice(unlabeled_indices, size=sample_size, replace=False)
    else:
        selected_indices = unlabeled_indices

    # 2. 将选定的无标签样本做成一个小的 DataLoader
    subset = torch.utils.data.Subset(dataset, selected_indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

    teacher_net.eval()  # Teacher 只推理

    all_features = []
    all_pseudo_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch  # label 应该是 -1
            images = images.to(device)

            # Teacher在 aug1 和 aug2 的推理，这里为了可视化简单，我们就只做一次普通推理
            # 如果你想和训练时严格一致，需要对同一张图片做2种增强再计算差异 diff。
            # 这里我们简化，只演示生成 pseudo_label & confidence。

            logits_t = teacher_net(images)  # shape=[B, num_classes]
            p = F.softmax(logits_t, dim=1)  # 概率

            # 简单地把 pseudo_label 设为 argmax(p)
            pseudo_label = p.argmax(dim=1)

            # 如果想用与训练一致的 "差异 -> w" 方式，需要我们再做一次不同增强，然后计算 diff = (p1-p2)...
            # 这里只演示最基础的 "置信度 = p.max(dim=1)[0]"。
            confidence = p.max(dim=1)[0]  # 取最大值作为置信度

            # ====== 如果想和训练时的 w 逻辑一致，举个示例： ======
            # aug2 = <做另一种增强的同一批图像>
            # logits_t2 = teacher_net(aug2)
            # p2 = F.softmax(logits_t2, dim=1)
            # diff = (p - p2).pow(2).sum(dim=1).sqrt()
            # w = torch.exp(-alpha * diff)
            # pseudo_label = 0.5*(p + p2).argmax(dim=1)
            # confidence = w  # 或者保存 w, 也行
            # =========================================

            # 提取特征
            features = get_features(teacher_net, images, layer_name)  # shape=[B, C, H, W]
            # 这里简单地做一个全局平均池化
            # 如果是resnet18 layer4, shape=[B, 512, 7, 7], GAP后 => [B, 512]
            features_gap = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

            all_features.append(features_gap.cpu().numpy())
            all_pseudo_labels.append(pseudo_label.cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())

    # 整合
    all_features = np.concatenate(all_features, axis=0)  # shape=[N, C]
    all_pseudo_labels = np.concatenate(all_pseudo_labels, 0)  # shape=[N]
    all_confidences = np.concatenate(all_confidences, 0)  # shape=[N]

    # 3. 用 t-SNE 把高维降到 2D
    print("Running t-SNE, this may take a while ...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    X_2d = tsne.fit_transform(all_features)

    # 4. 绘图
    plt.figure(figsize=(8, 6))

    # 因为有 200 个类别（CUB-200），如果全部都不一样颜色，可能比较乱。
    # 这里仅作演示，可以随机给每个类别一个颜色，或者只画一个散点图用 colormap。

    # 散点的颜色取 pseudo_label，但要小心 pseudo_label 值域(0~199)，需给一个colormap
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=all_pseudo_labels,
        s=50,
        cmap='tab20',  # 这个colormap一次最多支持20种明显颜色，再多会重复
        alpha=0.7
    )
    plt.colorbar(scatter, label='Pseudo-label (ID)')  # 颜色表示伪标签ID

    # 我们也可以用点的大小或透明度来表示置信度
    # 例如，如果想用点大小表示置信度：
    #  idx = np.argsort(all_confidences)  # 使低置信度的点先画，避免被高置信度点覆盖
    #  plt.scatter(
    #      X_2d[idx, 0],
    #      X_2d[idx, 1],
    #      c=all_pseudo_labels[idx],
    #      s=20 + 100*all_confidences[idx],  # 置信度越高点越大
    #      cmap='tab20',
    #      alpha=0.7
    #  )

    plt.title("t-SNE of Unlabeled Samples (Teacher's Pseudo-labels)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

    # 如果你想把图保存成文件
    # plt.savefig("pseudo_label_tsne.png")

