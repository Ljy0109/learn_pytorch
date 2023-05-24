import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

# test_set = datasets.CIFAR10("DataSet/CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor())

# test_dataset = DataLoader(test_set, batch_size=64, shuffle=True)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        # self.maxPooling1 = nn.MaxPool2d(2)
        # self.conv2 = nn.Conv2d(32, 32, 5, stride=2, padding=2)
        # self.maxPooling2 = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        # self.maxPooling3 = nn.MaxPool2d(2)
        # self.flatten1 = nn.Flatten()
        # self.liner1 = nn.Linear(1024, 64)
        # self.liner2 = nn.Linear(64, 10)
        # 下列Sequential与上面的部分是等效的，可以直接调用x = self.model1(x)
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
print(myNet)
# 检验网络正确性，即是否按照预想的过程进行，可以定义个全是1的数组作为输入
input = torch.ones((64, 3, 32, 32))
output = myNet(input)
# 可视化网络结构
writer = SummaryWriter("logs/nn_seq")
writer.add_graph(myNet, input)
writer.close()


