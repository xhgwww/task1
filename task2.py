import torch
import torchvision
import matplotlib.pyplot as plt
import pickle


# 定义loadImageData函数，用于读取手写数字图片数据，并进行训练集和测试集的划分
def loadImageData():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    sample_image, _ = trainset[89]  # 获取训练集中的第一张图片
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    print(sample_image.shape)  # 打印张量的形状

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    return trainloader, testloader


# 对训练样本和测试样本使用统一的均值、标准差进行归一化
def normalize(x, mean, std):
    x_normed = (x - mean) / std
    return x_normed


# 定义sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# 定义cost函数
def cost(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


# 定义两层的神经网络模型
class ANN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = sigmoid(x)
        x = self.fc2(x)
        x = sigmoid(x)
        return x


# 主函数部分
if __name__ == '__main__':
    # 载入图片数据
    trainloader, testloader = loadImageData()

    # 初始化设置
    input_size = 784
    hidden_size = 256
    output_size = 10
    lambda_reg = 0.01

    # 初始化网络模型
    model = ANN(input_size, hidden_size, output_size)

    # 设置学习率、循环次数等超参数
    learning_rate = 0.1
    num_epochs = 20
    train_loss = []
    test_acc = []

    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # 进行ANN的训练和测试，记录每轮的损失和准确率
    best_acc = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in trainloader:
            # 数据预处理部分
            X = X.view(X.shape[0], -1)
            Y = torch.zeros(X.shape[0], output_size)
            Y.scatter_(1, y.unsqueeze(1), 1)

            X = normalize(X, torch.mean(X), torch.std(X))

            # 前向传播
            outputs = model(X)

            # 计算损失
            loss = criterion(outputs, Y)
            epoch_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(epoch_loss / len(trainloader))

        with torch.no_grad():
            acc = 0
            total = 0
            for X, y in testloader:
                X = X.view(X.shape[0], -1)
                X = normalize(X, torch.mean(X), torch.std(X))
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                acc += (predicted == y).sum().item()
            acc = acc / total
            test_acc.append(acc)

            if acc > best_acc:
                torch.save(model.state_dict(), 'best_model.pt')
                best_acc = acc

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss[-1]}, Test Acc: {test_acc[-1]}')

    # 绘制损失和准确率的曲线图，标注最大准确率
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_acc, label='Test Accuracy')
    plt.axhline(y=best_acc, color='r', linestyle='--', label='Best Accuracy')
    plt.legend()
    plt.show()
