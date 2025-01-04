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
    transforms.ToTensor(),         # 转换为张量
])
test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor(),])


train_data = CUB_200(root='CUB-200', download=True, transform=train_transform)
test_data = CUB_200(root='CUB-200', download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

#model define
model = models.resnet50(weights=None)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#forward and
for epoch in tqdm(range(100)):
    for batch_images, batch_labels in train_loader:
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_images)

        # 计算损失
        loss = criterion(outputs, batch_labels)

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()