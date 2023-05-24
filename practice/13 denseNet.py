"""
自行实现的DenseNet模块
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = torchvision.datasets.CIFAR10(root="dataset/CIFAR10", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root="dataset/CIFAR10", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)

def conv_block(input_channels, output_channels):
    # DenseNet使用的是ResNet改良后的架构
    # 即，BN批量归一化层、激活层、卷积层
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, nums_conv, input_channels, output_channels):
        """
        :param nums_conv: 稠密层中的卷积层个数
        :param input_channels: 稠密层的输入通道数
        :param output_channels: 稠密层中的每个卷积层输出通道数都是固定的，均为output_channels
        """
        super().__init__()
        layer = []  # 用于存储稠密层的list
        for i in range(nums_conv):
            layer.append(conv_block(
                input_channels + output_channels * i, output_channels)
            )
        # 将layer解压到nn.Sequential中
        self.denseBlock = nn.Sequential(*layer)

    def forward(self, input):
        for block in self.denseBlock:
            output = block(input)
            # 在通道维度进行拼接
            input = torch.cat((input, output), dim=1)
        return input

def transition_block(input_channels, output_channels):
    # 稠密层的输出通道数会非常多，通道数过多会增加模型的复杂度
    # 因此使用过度模块降低通道数，内核是使用1x1卷积来降低通道数
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

# DenseNet在开始阶段使用了和ResNet相同的单卷积层和最大池化层
block1 = nn.Sequential(  # 输入图像为单通道的灰度图像，数据集使用为MNIST
    # MNIST图像28x28size太小了，改用三通道的CIFAR
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 这里的input指的是接下来输入的通道数，也就是block1输出的通道数
# 增长率指的是稠密层中每层卷积层输出的通道数
input_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 在每个稠密层中卷积层的个数
blks = []  # 存储层顺序的列表
for i, num_conv in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_conv, input_channels, growth_rate))
    # 计算上一个稠密层的输出通道数
    input_channels = input_channels + num_conv * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        # 在稠密层中间添加过渡层
        blks.append(transition_block(input_channels, input_channels // 2))
        input_channels = input_channels // 2

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            block1,
            *blks,
            nn.BatchNorm2d(input_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_channels, 10)
        )

    def forward(self, input):
        return self.model(input)

myNet = DenseNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myNet = myNet.to(device)

myLoss = nn.CrossEntropyLoss()
myOptim = torch.optim.SGD(myNet.parameters(), lr=0.01)

writer = SummaryWriter("logs/DenseNet")
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
    input = torch.ones((64, 3, 96, 96))
    input = input.to(device)
    writer.add_graph(myNet, input)
    for epoch in range(10):
        train(epoch)
        test(epoch)

    writer.close()