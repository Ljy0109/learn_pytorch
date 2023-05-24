"""
对于嵌套块的测试，以及对于参数访问和初始化的测试
"""
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
x = torch.randn(size=(2, 4))
net(x)
print("net[2] = {}\n".format(net[2]))
print("(type1) parameters of net[2] = {}\n".format(net[2].state_dict()))
print("(type2) parameters of net[2] = {}\n".format(net[2]._parameters))

print("net[2].bias = {}\n".format(net[2].bias))
# 使用data访问，数据形式是tensor型的，可以用于润色
print("net[2].bias.data = {}\n".format(net[2].bias.data))
# 使用item（）访问，只能返回单个元素，对于非标量的tensor不能使用
print("net[2].bias.item() = {}\n".format(net[2].bias.item()))
print("参数的梯度状态 net[2].weight.grad == None = {}\n".format(net[2].weight.grad == None))

# 访问一层的参数 *的作用应该是去除一层括号，或者说解压？(将数组等容器里的元素全部取出来）
print("net[0].named_parameters() = \n", *[(name, param.shape) for name, param in net[0].named_parameters()])
print("net.named_parameters() = \n", *[(name, param.shape) for name, param in net.named_parameters()])
# 参数是以字典的形式存储的，所以可以直接用字典访问
print("2.bias = {}".format(net.state_dict()['2.bias'].data))

# 嵌套块的操作方法
def Block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )

def Block2():
    net = nn.Sequential()  # 可以理解为将层按照字典的形式存储
    for i in range(4):
        # 嵌套操作 .add_module
        net.add_module(f'block {i}', Block1())
    return net

renet = nn.Sequential(Block2(), nn.Linear(4, 1))
print("nestBlock :\n{}".format(renet))

# 访问renet的第一个子块的第二个子块的第一层
renet[0][1][0].bias.data

# 参数初始化设置，该参数初始化函数可以自定义，使用方法相同
def init_normal(m):
    if type(m) == nn.Linear:
        # 如果是设置为常数则使用 nn.init.constant_(m.weight, 42)
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 设置为均值为0，标准差为0.01的正态分布
        nn.init.zeros_(m.bias)
net.apply(init_normal)  # 可以整个网络满足要求的层都使用
net[0].apply((init_normal))  # 也可以指定某层使用
net[0].weight.data[0, 0] = 42  # 也可以直接赋值

# 参数共享操作
# 将需要进行共享参数的层提供一个名称
shared_layer = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared_layer,
                    nn.ReLU(),
                    shared_layer,
                    nn.ReLU(),
                    nn.Linear(8, 1)
)
# 此时，net中的第三层和第五层是参数共享的，一层改变另一层也会变
# 参数共享的同时，梯度也是共享的，所以反向传播时，这两个层的梯度会加在一起
# 每层的梯度计算仍然是独立的，因为梯度下降 w = w - lr*grad
# 所以参数共享层的梯度下降是累加的
