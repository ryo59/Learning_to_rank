# -- coding: utf-8 --
import torch
import torch.nn as nn

def activation(str):
    if str == 'ReLU':
        return nn.ReLU() # ReLU(x)=max(0,x)
    elif str == 'LReLU':
        return nn.LeakyReLU(negative_slope=0.01)   # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
    elif str == 'ReLU6':
        return nn.ReLU6()   # ReLU6(x)=min(max(0,x),6)
    elif str == 'Sig':
        return nn.Sigmoid() # Sigmoid(x)= 1 / 1+exp(−x)
    elif str == 'Tan':
        return nn.Tanh()    # (exp(x)-exp(-x)) / (exp(x)+exp(-x))

    else:
        raise NotImplementedError

