"""
softmax层是用于多分类问题，考虑到类与类之间应该是具有相关性（相互抑制）的
所以不能对每一类使用sigmoid判断概率
softmax的结果有两个性质：
1. P(x = i) >= 0
2. sum (P(x = i)) = 1
为了满足上述性质，于是将概率设置为P(y = i) = (e^(z_i)) / sum(i)(e^(z_i))
Loss(Y_pred, Y) = sum (-Y * log Y_pred)
CrossEntropyLoss = Softmax + Log + NLLLoss
"""
import torch

myLoss = torch.nn.CrossEntropyLoss()
Y = torch.LongTensor([2, 0, 1])

Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])
Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])
l1 = myLoss(Y_pred1, Y)
l2 = myLoss(Y_pred2, Y)
print("Batch Loss1 = {}, Batch Loss2 = {}".format(l1 ,l2))