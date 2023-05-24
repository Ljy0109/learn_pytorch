"""
反向传播和优化器使用
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
optim = torch.optim.SGD(myNet.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0
    for data in test_dataset:
        imgs, targets = data
        output = myNet(imgs)
        result_loss = loss(output, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)
