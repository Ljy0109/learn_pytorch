"""
transformer中的加法&规范化实现
规范化使用的是layer normalization
AddNorm类主要使用残差链接、dropout和层归一化来实现
"""

import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # X和Y的尺寸应该是一样的
        # Y是前馈网络或多头注意力的输出
        return self.ln(self.dropout(Y) + X)


if __name__ == '__main__':
    add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    tmp = add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4)))
    print(tmp)
