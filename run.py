import torchvision.models as models
import torchvision.datasets as datasets
from writing_custom_datasets import CUB_200
import torchvision.transforms as transforms
import torch.utils.data
from tqdm import tqdm
import torch

# data processing
# transform
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将所有图片调整为 256x256
    transforms.ToTensor(),  # 转换为张量
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_data = CUB_200(root='CUB-200', download=True, transform=train_transform)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=5, prefetch_factor=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=5, prefetch_factor=2)

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

    ######################
    # validate the model #
    ######################
    model.eval()
    running_valid_loss = 0.0
    valid_batch_count = 0

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch_images)

            # calculate the batch loss
            loss = criterion(output, batch_labels)
            print(loss.item())

            # update running validation loss
            running_valid_loss += loss.item()
            valid_batch_count += 1

        # calculate average validation loss
        avg_valid_loss = running_valid_loss / valid_batch_count
        valid_losses.append(avg_valid_loss)
        print(avg_valid_loss)

    # print training/validation statistics
    print(f'Epoch: {epoch + 1} \tTraining Loss: {avg_train_loss:.6f} \tValidation Loss: {avg_valid_loss:.6f}')

