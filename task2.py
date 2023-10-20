import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 定义loadImageData函数，用于读取手写数字图片数据，并进行训练集和测试集的划分
def loadImageData():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader

# 对训练样本和测试样本使用统一的均值、标准差进行归一化
def normalize(x, mean, std):
    x_normed = (x - mean) / std
    return x_normed

# 定义OneHotEncoder函数，对输出标签进行独热编码
def OneHotEncoder(y):
    y_onehot = torch.zeros(y.shape[0], 10)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot

# 定义sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# 定义cost函数
def cost(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)

# 定义trainANN函数，用于训练一轮ANN，包括前向传播和反向传播过程，更新权重和偏置。
def trainANN(X, Y, W1, b1, W2, b2, lambda_reg, learning_rate):
    # 前向传播
    Z1 = torch.matmul(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = torch.matmul(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    # 反向传播
    dZ2 = A2 - Y
    dW2 = torch.matmul(A1.T, dZ2) / X.shape[0] + lambda_reg * W2
    db2 = torch.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
    dA1 = torch.matmul(dZ2, W2.T)
    dZ1 = dA1 * (sigmoid(Z1) * (1 - sigmoid(Z1)))
    dW1 = torch.matmul(X.T, dZ1) / X.shape[0] + lambda_reg * W1
    db1 = torch.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
    
    # 更新权重和偏置
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    
    return W1, b1, W2, b2

# 定义predictionANN函数，用于对输入数据进行预测
def predictANN(X, W1, b1, W2, b2):
    Z1 = torch.matmul(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = torch.matmul(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2

# 定义computeAcc函数，用于计算模型的准确率
def computeAcc(X, Y, W1, b1, W2, b2):
    Y_pred = predictANN(X, W1, b1, W2, b2)
    acc = (torch.argmax(Y_pred, axis=1) == torch.argmax(Y, axis=1)).float().mean()
    return acc

# 主函数部分
if __name__ == '__main__':
    # 载入图片数据
    trainloader, testloader = loadImageData()

    # 初始化设置
    input_size = 784
    hidden_size = 256
    output_size = 10
    lambda_reg = 0.01

    # 初始化网络参数
    W1 = torch.randn(input_size, hidden_size) / np.sqrt(input_size)
    b1 = torch.zeros(1, hidden_size)
    W2 = torch.randn(hidden_size, output_size) / np.sqrt(hidden_size)
    b2 = torch.zeros(1, output_size)

    # 设置学习率、循环次数等超参数
    learning_rate = 0.1
    num_epochs = 20
    train_loss = []
    test_acc = []

    # 进行ANN的训练和测试，记录每轮的损失和准确率
    best_acc = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in trainloader:
            # 数据预处理部分
            X = X.view(X.shape[0], -1)
            Y = OneHotEncoder(y)

            X = normalize(X, torch.mean(X), torch.std(X))

            # 训练部分
            W1, b1, W2, b2 = trainANN(X, Y, W1, b1, W2, b2, lambda_reg, learning_rate)

            loss = cost(predictANN(X, W1, b1, W2, b2), Y)
            epoch_loss += loss.item()

        train_loss.append(epoch_loss / len(trainloader))

        with torch.no_grad():
            acc = computeAcc(normalize(torch.cat([X for X, _ in testloader]), torch.mean(X), torch.std(X)),
                             OneHotEncoder(torch.cat([y for _, y in testloader])),
                             W1, b1, W2, b2)
            test_acc.append(acc.item())

            if acc.item() > best_acc:
                pickle.dump((W1, b1, W2, b2), open('best_model.pkl', 'wb'))
                best_acc = acc.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss[-1]}, Test Acc: {test_acc[-1]}')

    # 绘制损失和准确率的曲线图，标注最大准确率
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_acc, label='Test Accuracy')
    plt.axhline(y=best_acc, color='r', linestyle='--', label='Best Accuracy')
    plt.legend()
    plt.show()
