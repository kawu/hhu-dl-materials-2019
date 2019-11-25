from typing import Tuple

import torch
from torch.autograd import Function

# Tensor type, as usual
TT = torch.TensorType


################################################################
# Addition
################################################################


class Addition(Function):

    @staticmethod
    def forward(ctx, x: TT, y: TT) -> TT:
        return x + y
        
    @staticmethod
    def backward(ctx, dzdy: TT) -> Tuple[TT, TT]:
        return dzdy, dzdy


# Make `add` an alias of the custom autograd function
add = Addition.apply

# We can check that our custom addition behaves as the one provided in PyTorch.
# First, create two one-element tensors with 1.0 and 2.0, and then perform the
# backward computation (which means that the objective is to minimize/maximize 
# their sum).
x1 = torch.tensor(1.0, requires_grad=True)
y1 = torch.tensor(2.0, requires_grad=True)
(x1 + y1).backward()

# We do the same with our custom addition function. 
x2 = torch.tensor(1.0, requires_grad=True)
y2 = torch.tensor(2.0, requires_grad=True)
add(x2, y2).backward()

assert x1.grad == x2.grad
assert y1.grad == y2.grad
