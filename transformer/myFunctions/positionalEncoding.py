"""
实现位置编码
"""
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        # dropout是一个阈值，随机生成一组0~1的数，返回大于阈值的索引
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        # i / (10000^(2j/d))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # 0::2 表示从索引0开始，以步长为2选择元素，即选择索引为偶数的列
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == '__main__':
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]

    pass
