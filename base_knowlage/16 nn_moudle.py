from torch import nn
import torch

class Modle(nn.Moudle):
    def __init__(self):
        super().__init__()

    def forward(self, imput):
        output = input + 1
        return output

modle = Modle()
x = torch.tensor(1.0)
output = modle.forward(x)
print(output)