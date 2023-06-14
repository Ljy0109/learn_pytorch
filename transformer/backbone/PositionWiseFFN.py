"""
基于位置的前馈网络实现
即多层感知机，两个全连接层，一个relu层
"""
import torch
import torch.nn as nn


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 如果输入没有展平的话，全连接层只会对最内层数据进行处理
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


if __name__ == '__main__':
    ffn = PositionWiseFFN(4, 4, 8)
    ffn.eval()
    tmp = ffn(torch.ones((2, 3, 4)))[0]
