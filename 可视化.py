import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
data = pd.read_csv("C:\\Users\\hlc\\Desktop\\jotang\\Task 1+\\train_data .csv")

# 散点图
def plot_scatter(x, y):
    plt.figure()
    for label in data["label"].unique():
        plt.scatter(data[data["label"]==label][x], data[data["label"]==label][y], label=label, s=1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs. {y}")
    plt.legend()
    plt.show()

# 三维散点图
def plot_3d_scatter(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in data["label"].unique():
        ax.scatter(data[data["label"]==label][x], data[data["label"]==label][y], data[data["label"]==label][z], label=label, s=1)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend()
    plt.show()


# 绘图

plot_scatter("feature3","feature9")
plot_scatter("feature12","feature30")
plot_scatter("feature19","feature102")
plot_3d_scatter("feature18", "feature38", "feature79")

plt.show()