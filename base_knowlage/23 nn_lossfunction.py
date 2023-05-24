"""
损失函数和反向传播
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

test_set = datasets.CIFAR10("DataSet/CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor())

test_dataset = DataLoader(test_set, batch_size=64, shuffle=True)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self, input):
        output = self.model1(input)
        return output

myNet = MyNet()
loss = nn.CrossEntropyLoss()
for data in test_dataset:
    imgs, targets = data
    output = myNet(imgs)
    result_loss = loss(output, targets)
    result_loss.backward()
    # 反向梯度的位置在myNet/model1/protect//modles/'0'/weight/grad

