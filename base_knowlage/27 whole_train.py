import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10("DataSet/CIFAR10_dataset", train=True, transform=torchvision.transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10("DataSet/CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor())

# 查看数据集的长度 len()
len_train = len(train_set)
len_test = len(test_set)
print("训练集和测试集的长度分别为{}和{}".format(len_train, len_test))

train_dataset = DataLoader(train_set, batch_size=64, shuffle=True)
test_dataset = DataLoader(test_set, batch_size=64, shuffle=True)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()  # 初始化父类
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output

device = torch.device("cuda:0")
myNet = MyNet()
myNet = myNet.to(device)
# 使用一个样例对网络结构进行测试
# test_sample = torch.ones((64, 3, 32, 32))
# test_output = myNet(test_sample)


writer = SummaryWriter("logs/train")
train_step = 0
test_step = 0
epoch = 10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(myNet.parameters(), lr=0.005)

for i in range(epoch):
    print("------------------当前训练轮数：{}------------------".format(i + 1))

    # 训练步骤开始 每个batch的训练
    # myNet.train() 只对Dropout 和 BatchNorm有作用
    for data in train_dataset:
        optimizer.zero_grad()
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = myNet.forward(imgs)
        train_loss = loss(output, targets)
        train_loss = train_loss.to(device)
        # 优化器调优
        train_loss.backward()
        optimizer.step()
        train_step += 1
        if train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(train_step, train_loss.item()))
            writer.add_scalar("train_100", train_loss, train_step)

    # 测试步骤开始 with代码块内设置torch不使用梯度，也就是不训练
    # myNet.eval() 只对Dropout 和 BatchNorm有作用
    total_test_loss = 0
    test_step += 1
    total_precision = 0
    with torch.no_grad():
        for data in test_dataset:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = myNet(imgs)
            # output的每行是对一个图像属于哪一类的概率
            # argmax(0)是返回每列的最大值，argmax(1)是返回每行的最大值
            precision = (output.argmax(1) == targets).sum()
            total_precision += precision
            test_loss = loss(output, targets)
            test_loss = test_loss.to(device)  # ?
            total_test_loss += test_loss

    print("整个训练集的Loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, test_step)
    print("整个训练集的Precision：{}".format(total_precision / len_test))
    writer.add_scalar("test_precision", total_precision / len_test, test_step)

    torch.save(myNet.state_dict(), "model/myNet_{}.pth".format(i))
    print("第{}轮模型已保存".format(i))

writer.close()

