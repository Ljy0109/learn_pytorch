"""
logistic function: sigma(x) = 1 / (1 + e^(-x)) 在[-1, 1]区间内 sigmoid函数
该函数是饱和函数，该函数会随着x的增加或减少而发生梯度下降并趋近与0
loss = -(y * log(y_pred) + (1-y)log(1-y_pred))
因为y_pred表示的是概率，所以loss计算的是概率分布的差异
计算分布差异的方式有KL散度和交叉熵( sum_i (P_D(x = i) * ln(P_T(x = i)) 越大越好)
"""

"""
与linear regression的差异在于
1. import torch.nn.functional as F
2. y_pred = F.sigmoid(self.linear(x))
3. loss = nn.BCELoss(size_average=False)  # BCE 二分类的cross-entropy
"""
