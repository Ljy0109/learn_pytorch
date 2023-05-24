"""
GoogleNet的层中层inception,为了避免代码重复，应该新定义一个类来封装
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

train_dataset = torchvision.datasets.MNIST(root="dataset/MNIST", train=True, download=True, transform=torchvision.transforms.ToTensor())

test_dataset = torchvision.datasets.MNIST(root="dataset/MNIST", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)

class Inception(nn.Module):
    """
    inception中有4个分支，分别是池化分支、1×1卷积、5×5卷积、3×3卷积
    """
    def __init__(self, in_channels):
        super().__init__()
        # 池化分支 平均池化 + 1×1卷积（输出24通道）
        # 池化不会影响通道数，但需要设置步长和padding来保证图像尺寸不变
        # 卷积会影响通道数，但同样需要设置步长和padding来保证图像尺寸不变
        self.branch_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_2 = nn.Conv2d(in_channels, 24, kernel_size=1)
        # 1×1卷积分支 输出16通道
        self.branch_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        # 5×5卷积分支 1×1卷积（输出16通道） + 5×5卷积（输出24通道）
        self.branch_5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_5x5_2 = nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2)
        # 3×3卷积分支 1×1卷积（输出16通道） + 3×3卷积（输出24通道） + 3×3卷积（输出24通道）
        self.branch_3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_3x3_2 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1)
        self.branch_3x3_3 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        branch_pool = self.branch_pool_1(input)
        branch_pool = self.branch_pool_2(branch_pool)

        branch_1x1 = self.branch_1x1(input)

        branch_5x5 = self.branch_5x5_1(input)
        branch_5x5 = self.branch_5x5_2(branch_5x5)

        branch_3x3 = self.branch_3x3_1(input)
        branch_3x3 = self.branch_3x3_2(branch_3x3)
        branch_3x3 = self.branch_3x3_3(branch_3x3)

        outputs = [branch_pool, branch_1x1, branch_5x5, branch_3x3]
        return torch.cat(outputs, dim=1)

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # inception的输出通道数是88
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = Inception(in_channels=10)
        self.incep2 = Inception(in_channels=20)

        self.maxpooling = nn.MaxPool2d(2)
        # 对展平后的特征做线性层（全连接）
        # 获得1408这个数的方法是直接运行一次，查看输出
        self.linear1 = nn.Linear(1408, 10)
        self.flatten1 = nn.Flatten()

    def forward(self, input):
        output = self.conv1(input)
        output = self.maxpooling(output)
        output = nn.functional.relu(output)

        output = self.incep1(output)

        output = self.conv2(output)
        output = self.maxpooling(output)
        output = nn.functional.relu(output)

        output = self.incep2(output)

        output = self.flatten1(output)
        output = self.linear1(output)
        return output

myNet = MyNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myNet = myNet.to(device)

myOptim = torch.optim.SGD(myNet.parameters(), lr=0.005)
myLoss = nn.CrossEntropyLoss()

writer = SummaryWriter("logs/inception")  # 查看时，需要输入从父文件开始的路径
train_step = 0
def train(epoch):
    global train_step
    for data in train_data:  # 五步走
        # 梯度清零
        myOptim.zero_grad()
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 预测
        outputs = myNet.forward(imgs)
        # 计算损失
        train_loss = myLoss(outputs, targets)
        # train_loss = train_loss.to(device)
        # 反向传播
        train_loss.backward()
        # 更新
        myOptim.step()

        train_step += 1
        if (train_step % 100 == 0):
            print("[当前训练轮数 = {}， 训练次数 = {}]， Loss = {:.5f}".format(epoch + 1, train_step, train_loss.item()))
            writer.add_scalar("train", train_loss.item(), train_step)

def test(epoch):
    total_precision = 0
    with torch.no_grad():
        for data in test_data:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = myNet.forward(imgs)
            # output的每行是对一个图像属于哪一类的概率
            # argmax(0)是返回每列的最大值，argmax(1)是返回每行的最大值
            precision = (output.argmax(1) == targets).sum()
            total_precision += precision
            # test_loss = myLoss(output, targets)
        print("Accuracy on test set: {}%".format(100 * (total_precision / len(test_data.dataset))))
        writer.add_scalar("test", 100 * (total_precision / len(test_data.dataset)), epoch + 1)


if __name__ == '__main__':
    input = torch.ones((64, 1, 28, 28))
    input = input.to(device)
    writer.add_graph(myNet, input)
    for epoch in range(10):
        train(epoch)
        test(epoch)

    writer.close()

