"""
缩放点积注意力机制实现
"""
import torch
import torch.nn as nn
import math

import sys
sys.path.append(r'myFunctions')

from masked_softmax import masked_softmax

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]  # 查询和键的长度，要求必须相等
        # 设置transpose_b=True为了交换keys的最后两个维度
        # 使keys的形状转置为(batch_size，d，“键－值”对的个数)
        # scores的形状为(batch_size，查询的个数，“键－值”对的个数)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 模型为评估模式时，dropout不会起作用
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    tmp = attention(queries, keys, values, valid_lens)
    print(tmp)
