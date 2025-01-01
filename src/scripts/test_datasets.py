from PIL import Image
from torch.utils.data import DataLoader
from src.data.writing_custom_datasets import CUB_200
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义数据预处理（可以考虑调整填充颜色）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 建议使用224，这是许多预训练模型的标准输入尺寸
    transforms.ToTensor(),
    transforms.RandomRotation(30, fill=(255, 255, 255)),  # 使用白色填充
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def visualize_image_and_pixels(dataset, transform):
    # 选择一个图像
    image_path = dataset.filtered_paths[0]  # 选择第一张图像
    label = dataset.filtered_labels[0]
    class_name = dataset.classes[label - 1]  # 注意类别索引偏移

    # 读取原始图像
    original_image = Image.open(dataset.root / dataset.base_folder / 'images' / image_path).convert('RGB')
    # 应用变换
    transformed_image = transform(original_image)

    # 创建更大的图形
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # 增大图形尺寸
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image\nClass: {class_name}', fontsize=16)
    axes[0].axis('off')

    # 处理归一化和变换后的图像显示
    transformed_pixels = transformed_image.numpy()

    # 对像素值进行截断和缩放，使其在可显示范围内
    transformed_pixels_display = np.clip(transformed_pixels, -1, 1)
    transformed_pixels_display = (transformed_pixels_display + 1) / 2  # 将范围从[-1,1]映射到[0,1]

    # 减少网格密度，使文字更清晰
    step = max(transformed_pixels.shape[1:]) // 10  # 减少网格数量

    axes[1].imshow(transformed_pixels_display.transpose(1, 2, 0))
    axes[1].set_title('Transformed & Normalized Pixel Values', fontsize=16)
    axes[1].axis('off')

    # 在归一化和变换后的图像上显示像素值
    for i in range(0, transformed_pixels.shape[1], step):
        for j in range(0, transformed_pixels.shape[2], step):
            # 获取每个通道的原始像素值
            r_val = transformed_pixels[0, i, j]
            g_val = transformed_pixels[1, i, j]
            b_val = transformed_pixels[2, i, j]

            axes[1].text(j, i, f'R:{r_val:.3f}\nG:{g_val:.3f}\nB:{b_val:.3f}',
                         color='white',
                         fontsize=10,  # 增大字体大小
                         bbox=dict(facecolor='black', alpha=0.6, boxstyle='round'))

    # 打印一些额外信息
    print(f"Original Image Shape: {np.array(original_image).shape}")
    print(f"Transformed Image Shape: {transformed_image.shape}")
    print(f"Transformed Image Pixel Value Range:")
    print(f"Min: {transformed_pixels.min()}, Max: {transformed_pixels.max()}")

    plt.tight_layout(pad=3)  # 增加子图间距
    plt.show()


# 初始化数据集
dataset = CUB_200(
    root="CUB-200",
    train=True,  # 明确指定训练集
    transform=transform,
    download=True
)

# 打印类别和样本数量
print(f"Classes: {len(dataset.classes)}")
print(f"Samples: {len(dataset)}")

# 可视化单个图像及其像素
visualize_image_and_pixels(dataset, transform)

# 测试 DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(data_loader))

# 批量图像展示
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

for idx, (image, label) in enumerate(zip(images, labels)):
    # 反归一化
    img = image.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # 显示图像
    axes[idx].imshow(img)
    axes[idx].set_title(f'Class: {dataset.classes[label - 1]}')  # 注意类别索引
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# 测试 DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(data_loader))

# 计算行列数
num_images = 32
rows =4
cols = 8

# 创建大图形
fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
axes = axes.ravel()

for idx, (image, label) in enumerate(zip(images, labels)):
    # 反归一化
    img = image.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    # 显示图像
    axes[idx].imshow(img)
    axes[idx].set_title(f'Class: {dataset.classes[label - 1]}', fontsize=10)  # 注意类别索引
    axes[idx].axis('off')

plt.tight_layout(pad=0.5)  # 减小间距
plt.suptitle('Batch of 32 Images', fontsize=16, y=1.02)  # 添加总标题
plt.show()
