"""
当目标函数值在逐渐增大，说明训练发散了，即训练不收敛，可能是学习率太大
梯度下降方法不适用于具有鞍点的函数
鞍点：梯度为0的点，使用随机梯度来改善
随机梯度下降SGD：正常是使用所有样本的均值来计算梯度，随机是指随机选取一个样本求梯度
但是！ 随机梯度下降的权重之间更新是有依赖关系的，也就是w_(i)是由w_i计算得来
也就是说随机梯度下降不能并行计算，而普通梯度下降是可以并行的
一个折中的方法，取batch（一组样本）来计算，也就是批量随机梯度下降 Mini_batch
http://faculty.bicmr.pku.edu.cn/~wenzw/bigdata/matrix-cook-book.pdf
"""
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # 给梯度下降提供一个起点
alpha = 0.01  # 梯度下降的步长比率，学习率

def forward(x):
    return x * w

def myCost(x_data, y_data):
    # 目标函数MSE(均方误差) cost = (1/N) * sum( (y^_n - y_n)^2 )
    cost = 0
    for x, y in zip(x_data, y_data):
        cost += (forward(x) - y) ** 2
    return cost / len(x_data)
"""
def SGD(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
"""


def gradient(x_data, y_data):
    # 目标函数的梯度（对w求偏导）为 (1/N) * sum( 2 * (x_n * w - y_n) * x_n )
    grad = 0
    for x, y in zip(x_data, y_data):
        grad += 2 * (forward(x) - y) * x

    return grad / len(x_data)

print("Predict (before training)", 4, forward(4))
for epoch in range(100):
    cost = myCost(x_data, y_data)
    grad = gradient(x_data, y_data)
    w -= alpha * grad
    print("epoch = {}, w = {:.2f}, Predict(x = 4) = {:.2f}".format(epoch, w, forward(4)))

print("end of train")

