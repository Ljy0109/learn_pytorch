"""
resNet的残差模块，一共分两层
输入x，第一层激活后为F(x)，第二层激活后为H(x)=F(x)+x
这样的好处是使用H(x)对x求偏导时，始终为F'(x)+1
避免了F'(x)的梯度消失问题
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = torchvision.datasets.MNIST(root="dataset/MNIST", train=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="dataset/MNIST", train=False, transform=torchvision.transforms.ToTensor())

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)

class ResidualBlock(nn.Module):
    # 残差模块需要保证输入和输出的通道数是一样的(why?)
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, input):
        output = self.conv1(input)
        output = nn.functional.relu(output)  # F(x)
        output = self.conv2(output)
        output = nn.functional.relu(output + input)  # H(x) = F(x) + x
        return output

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)  # MNIST是单通道的灰度图像
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.res1 = ResidualBlock(16)  #
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res2 = ResidualBlock(32)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, 10)  # 线性层（全连接），将特征维度从512降到10分类

    def forward(self, input):
        output = self.conv1(input)
        output = nn.functional.relu(output)
        output = self.maxpooling(output)
        output = self.res1(output)
        output = self.conv2(output)
        output = nn.functional.relu(output)
        output = self.maxpooling(output)
        output = self.res2(output)
        output = self.linear1(self.flatten(output))
        return output

myNet = MyNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myNet = myNet.to(device)

myLoss = nn.CrossEntropyLoss()
myOptim = torch.optim.SGD(myNet.parameters(), lr=0.01)

writer = SummaryWriter("logs/resNet")
train_step = 0
def train(epoch):
    global train_step
    for data in train_data:
        myOptim.zero_grad()
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = myNet(imgs)
        loss = myLoss(outputs, targets)
        loss.backward()
        myOptim.step()
        train_step += 1
        if train_step % 100 == 0:
            print("[epoch = {}, train_step = {}], Loss = {}".format(epoch + 1, train_step, loss.item()))
            writer.add_scalar("train", loss.item(), train_step)

def test(epoch):
    with torch.no_grad():
        total_precision = 0
        for data in test_data:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = myNet(imgs)
            precision = (outputs.argmax(1) == targets).sum()
            total_precision += precision
        print("Accuracy on test dataset: {}".format(100 * (total_precision / len(test_data.dataset))))
        writer.add_scalar("test", 100 * (total_precision / len(test_data.dataset)), epoch + 1)

if __name__ == '__main__':
    input = torch.ones((64, 1, 28, 28))
    input = input.to(device)
    writer.add_graph(myNet, input)
    for epoch in range(10):
        train(epoch)
        test(epoch)

    writer.close()

