import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch
import time  # 添加time模块导入

# data processing
# transform
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_data = CUB_200(root='CUB-200', download=True, transform=train_transform)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform)

# 预加载数据到GPU
print("Preloading training data to GPU...")
start_time = time.time()
# 确保所有数据都加载到GPU
all_train_images = []
all_train_labels = []
for images, labels in tqdm(train_data):
    if isinstance(images, torch.Tensor):
        all_train_images.append(images.cuda())
    else:
        all_train_images.append(torch.tensor(images).cuda())
    all_train_labels.append(torch.tensor(labels).cuda())
train_data.images = torch.stack(all_train_images)
train_data.labels = torch.stack(all_train_labels)
print(f"Training data preloaded in {time.time() - start_time:.2f} seconds")

print("Preloading test data to GPU...")
start_time = time.time()
all_test_images = []
all_test_labels = []
for images, labels in tqdm(test_data):
    if isinstance(images, torch.Tensor):
        all_test_images.append(images.cuda())
    else:
        all_test_images.append(torch.tensor(images).cuda())
    all_test_labels.append(torch.tensor(labels).cuda())
test_data.images = torch.stack(all_test_images)
test_data.labels = torch.stack(all_test_labels)
print(f"Test data preloaded in {time.time() - start_time:.2f} seconds")

# 其余代码保持不变...


train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0, prefetch_factor=None)

# model define
criterion = torch.nn.CrossEntropyLoss()
model = models.resnet50(weights=None)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.fc = torch.nn.Linear(model.fc.in_features, 200)

# gpu
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()

# Initialize variables to track best model performance
best_val_loss = float('inf')
train_losses = []
valid_losses = []

print('test')


def validate(model, test_loader, train_on_gpu=True):
    # 初始化计数器
    valid_total = 0
    valid_correct = 0

    model.eval()  # 设置为评估模式

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            if train_on_gpu:
                batch_images = batch_images.cuda()
                batch_labels = batch_labels.cuda()

            output = model(batch_images)

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            valid_total += batch_labels.size(0)
            valid_correct += (predicted == batch_labels).sum().item()

        # Calculate validation accuracy
        valid_accuracy = 100 * valid_correct / valid_total

        return valid_accuracy


# forward and backward
for epoch in tqdm(range(100)):
    # Train mode
    model.train()
    running_train_loss = 0.0
    train_batch_count = 0

    for batch_images, batch_labels in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_images)

        # 计算损失
        loss = criterion(outputs, batch_labels)

        # 累加训练损失
        running_train_loss += loss.item()
        train_batch_count += 1

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

    # Calculate average training loss for this epoch
    avg_train_loss = running_train_loss / train_batch_count
    train_losses.append(avg_train_loss)
    model.eval()
    running_valid_loss = 0.0
    valid_batch_count = 0

    accuracy = validate(model, test_loader, train_on_gpu=True)
    print(f'Validation Accuracy: {accuracy:.2f}%')

    # print training/validation statistics
    print(f'Epoch: {epoch + 1} \tTraining Loss: {avg_train_loss:.6f} ')

