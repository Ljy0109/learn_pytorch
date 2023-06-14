import torch
import torch.nn as nn
import math


def sequence_mask(X, valid_lens, value=0):
    """生成掩蔽矩阵"""
    maxlen = X.size(1)  # 获取序列的最大长度
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_lens[:, None]  # 生成布尔类型的掩蔽矩阵
    mask = mask.type(torch.float32)  # 将布尔类型的掩蔽矩阵转换为浮点型
    return X * mask + value * (1 - mask)  # 将非掩蔽元素设为指定的value值


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 将valid_lens沿着第二个维度重复扩展，直到与X的形状一致
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)