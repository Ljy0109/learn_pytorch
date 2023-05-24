"""
MLP（多层感知机） 实现一个输入层，一个隐藏层，一个输出层
该代码试图不使用nn.Module来定义网络模型而是自己定义net(input)
"""
import torch
import torchvision.transforms.v2
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_dataset = datasets.MNIST(root="dataset/MNIST", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.MNIST(root="dataset/MNIST", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)
############################################################################
###############################关键步骤######################################

# 输入为28x28=784个灰度像素值，输出为10分类，单隐藏层共设置了256个隐藏单元
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# H = W1*X + b1 隐藏层参数
W1 = nn.Parameter(torch.randn(  # 生成服从标准正态分布的张量
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# output = W2*H + b2 输出层参数
W2 = nn.Parameter(torch.randn(  # 生成服从标准正态分布的张量
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
param = [W1, b1, W2, b2]

def relu(x):
    tmp = torch.zeros_like(x)  # 创建一个与x大小相同的全0张量
    return torch.max(x, tmp)

def net(input):  # 函数不能使用to(device),可以考虑定义类来实现
    # 该方法只能适用于batch_size=1的情况，大于1时，不能改变对参数W1尺寸的定义
    # 需要将batch分为单个矩阵，然后才能计算，不能整个batch进行计算
    input = torch.reshape(input, (-1, num_inputs)) # input变为行向量
    H = relu(input @ W1 + b1)  # @表示矩阵乘法
    return relu(H @ W2 + b2)

myLoss = nn.CrossEntropyLoss()
myOptim = torch.optim.SGD(param, lr=0.01)

############################################################################
############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)

writer = SummaryWriter("logs/MLP")
train_step = 0
def train(epoch):
    global train_step
    for data in train_data:
        myOptim.zero_grad()
        imgs, targets = data
        # imgs, targets = imgs.to(device), targets.to(device)
        outputs = net(imgs)
        loss = myLoss(outputs, targets)
        loss.backward()
        myOptim.step()
        train_step += 1
        if train_step % 100 == 0:
            print("[epoch = {}, train_step = {}], Loss = {}".format(epoch + 1, train_step, loss.item()))
            writer.add_scalar("train", loss.item(), train_step)

def test(epoch):
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data:
            imgs, targets = data
            # imgs, targets = imgs.to(device), targets.to(device)
            outputs = net(imgs)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
        print("Accuracy on test dataset: {}%".format(100 * (total_accuracy / len(test_data.dataset))))
        writer.add_scalar("test", 100 * (total_accuracy / len(test_data.dataset)), epoch)

if __name__ == "__main__":
    # input = torch.ones((64, 1, 28, 28))
    # input = input.to(device)
    # writer.add_graph(net, input)
    for epoch in range(10):
        train(epoch)
        test(epoch)

    writer.close()
