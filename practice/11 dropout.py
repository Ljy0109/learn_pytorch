"""
自行实现dropout模块
"""
import torch

def dropout_layer(x,dropout):
    if dropout == 1:
        return torch.zeros_like(x)
    if dropout == 0:
        return x
    mask = (torch.rand(x.shape) > dropout).float()
    return mask * x / (1 - dropout)
