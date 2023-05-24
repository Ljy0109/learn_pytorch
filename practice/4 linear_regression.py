"""
一个简单的线性模型的训练
"""
import torch

##################四步走######################
# 1. 准备数据集
# 2. 使用nn.Module设计模型
# 3. 构造损失函数和优化器
# 4. 构造训练循环
#############################################
# 内部每个[]是一行,每一行相当于一个样本，列数相当于特征维度
x_data = torch.tensor([[1.0],
                       [2.0],
                       [3.0]])
y_data = torch.tensor([[2.0],
                       [4.0],
                       [6.0]])

class linearModel(torch.nn.Module):
    # 重写父类函数
    def __init__(self):
        super(linearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # (输入特征的维度， 输出的维度)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = linearModel()
# 等于True则会求均值，一般来说没有什么影响，但是当batch的尺寸不一致时，应该令其为True
myLoss = torch.nn.MSELoss(size_average=False)  # loss = criterion(y_pred, y)
# model.parameters()是调用的Module的成员函数，该函数会model中的所有成员，并返回需要优化的参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr学习率，即梯度前面的权重系数

for epoch in range(100):  # 训练循环五步走：预测、损失、梯度清零、反向传播、更新
    y_pred = model(x_data)  # 计算预测值
    loss = myLoss(y_pred, y_data)  # 计算损失函数
    print("epoch = {}, loss = {:.2f}".format(epoch, loss.item()))

    optimizer.zero_grad()  # 梯度清零，否则梯度会被累加
    loss.backward()  # 反向传播
    optimizer.step()  # 更新

print("w = {:.2f}".format(model.linear.weight.item()))
print("b = {:.2f}".format(model.linear.bias.item()))

x_test = torch.tensor([4.0])
y_test = model(x_test)
print("predict(x = 4) = {:.2f}".format(y_test.item()))

