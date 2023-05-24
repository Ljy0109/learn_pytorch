"""
线性组合的叠加仍然是线性的，因此需要引入非线性激活函数
神经网络要找的就是特征的非线性空间变换函数
反向传播就是挨个计算每层的梯度值，注意，这里的梯度指的是损失函数对w的偏导
通过反向传播的梯度值进行batch随机梯度下降，并更新权重
"""
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])
# 对tensor变量，需要使用反向传播函数backward时，需要令requires_grad = True
w.requires_grad = True
alpha = 0.01  # 学习率

def forward(x):
    return x * w  # x自动转为tensor类型

def myLoss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 使用tensor构建计算图
        loss = myLoss(x, y)
        loss.backward()
        # w.grad是loss对w的偏导，.item()是直接取tensor中的数值
        print("x = {}, y = {}, w = {:2f}".format(x, y, w.grad.item()))
        # .data是取tensor中的数值来运算,权重更新不能直接使用tensor
        w.data -= alpha * w.grad.data
        # 把w中的梯度数据清零，如果不清零，那么会累加每次反向传播的梯度
        w.grad.data.zero_()

    print("processing: {}, {:2f}".format(epoch, loss.item()))

print("predict: f({}) = {:2f}".format(4, forward(4).item()))


