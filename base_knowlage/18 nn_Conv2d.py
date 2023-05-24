import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("DataSet/CIFAR10_dataset", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
    def forward(self, input):
        output = self.conv1(input)
        return output

myNet = MyNet()
writer = SummaryWriter("logs/nn_Conv2d_test")
step = 0
for data in dataloader:
    imgs, targets = data
    output = myNet.forward(imgs)
    writer.add_images("test", imgs, step)
    # 因为6通道数无法显示图像，所以使用torch.reshape来修改通道数
    output = torch.reshape(output, (-1, 3, 30, 30))
    # -1的位置可以根据其他尺寸的调整而自适应调整
    writer.add_images("con2d_test", output, step)
    step += 1

writer.close()
