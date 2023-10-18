import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 设置随机种子以便结果可复现
torch.manual_seed(42)

# 设定超参数
root_dir = "./data"
batch_size = 64
learning_rate = 0.001
epochs = 10

# 定义模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层和第一个池化层（下采样层） C1层、S2层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.ffn1 = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.BatchNorm1d(84),
            nn.ReLU()
        )

        # 输出层
        self.ffn3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10), 
            nn.BatchNorm1d(10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.ffn1(x)
        x = self.ffn2(x)
        x = self.ffn3(x)
        return x

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root=root_dir, train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=root_dir, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total:.2f}%')
    
