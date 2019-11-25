from typing import Tuple

import math
import torch
from torch.autograd import Function

# Tensor type, as usual
TT = torch.TensorType


################################################################
# Addition
################################################################


class Addition(Function):

    @staticmethod
    def forward(ctx, x1: TT, x2: TT) -> TT:
        return x1 + x2
        
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


# The nice part is that, since addition is element-wise, this should work also
# for complex tensors.
x1 = torch.randn(3, 3, requires_grad=True)
y1 = torch.randn(3, 3, requires_grad=True)
(x1 + y1).sum().backward()

# We use clone -> detach to get the exact copies of x1 and y1, otherwise
# not related to x1 and y1.
x2 = x1.clone().detach().requires_grad_(True)
y2 = y1.clone().detach().requires_grad_(True)
add(x2, y2).sum().backward()

# `x1.grad == x2.grad` creates a tensor of Boolean values, we use
# .all() to check that they are all True.
assert (x1.grad == x2.grad).all()
assert (y1.grad == y2.grad).all()


################################################################
# Sigmoid (logistic)
################################################################


class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x: TT) -> TT:
        y = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(y)
        return y
        
    @staticmethod
    def backward(ctx, dzdy: TT) -> TT:
        y, = ctx.saved_tensors
        return dzdy * y * (1 - y)


# Alias
sigmoid = Sigmoid.apply

# Some tests to check if this works as intended
x1 = torch.randn(3, 3, requires_grad=True)
torch.sigmoid(x1).sum().backward()

x2 = x1.clone().detach().requires_grad_(True)
sigmoid(x2).sum().backward()

# Check if sufficiently similar (clearly the backward method of the
# PyTorch sigmoid is better in terms of numerical precision).
diff = x1.grad - x2.grad
assert (-1e-5 < diff).all()
assert (diff  < 1e-5).all()


################################################################
# Sum
################################################################


class Sum(Function):

    @staticmethod
    def forward(ctx, x: TT) -> TT:
        ctx.save_for_backward(x)
        return x.sum()
        
    @staticmethod
    def backward(ctx, dzdy: TT) -> TT:
        x, = ctx.saved_tensors
        return torch.full_like(x, dzdy)

# Alias
tsum = Sum.apply

# Checks
x1 = torch.randn(1000, 1000, requires_grad=True)
torch.sum(x1).backward()

x2 = x1.clone().detach().requires_grad_(True)
tsum(x2).backward()

assert (x1.grad - x2.grad == 0).all()


################################################################
# Dot product
################################################################


# # Dot product (composition of PyTorch functions)
# def dot0(x: TT, y: TT) -> TT:
#     return torch.sum(x * y)


class Product(Function):

    @staticmethod
    def forward(ctx, x1: TT, x2: TT) -> TT:
        ctx.save_for_backward(x1, x2)
        return x1 * x2
        
    @staticmethod
    def backward(ctx, dzdy: TT) -> Tuple[TT, TT]:
        x1, x2 = ctx.saved_tensors
        return dzdy*x2, dzdy*x1

# Alias
prod = Product.apply

# Dot product (composition of custom functions)
def dot(x: TT, y: TT) -> TT:
    return tsum(prod(x, y))

# Checks
x1 = torch.randn(10, requires_grad=True)
y1 = torch.randn(10, requires_grad=True)
z1 = torch.dot(x1, y1)
z1.backward()

x2 = x1.clone().detach().requires_grad_(True)
y2 = y1.clone().detach().requires_grad_(True)
z2 = dot(x2, y2)
z2.backward()

diff = z1 - z2
assert (-1e-5 < diff).all()
assert (diff  < 1e-5).all()

assert (x1.grad - x2.grad == 0).all()
assert (y1.grad - y2.grad == 0).all()


################################################################
# Dot product in one pass
################################################################


class DotProduct(Function):

    @staticmethod
    def forward(ctx, x1: TT, x2: TT) -> TT:
        ctx.save_for_backward(x1, x2)
        return (x1 * x2).sum()
        
    @staticmethod
    def backward(ctx, dzdy: TT) -> Tuple[TT, TT]:
        x1, x2 = ctx.saved_tensors
        return dzdy*x2, dzdy*x1

# Alias
dot = DotProduct.apply

# Checks
x1 = torch.randn(10, requires_grad=True)
y1 = torch.randn(10, requires_grad=True)
z1 = torch.dot(x1, y1)
z1.backward()

x2 = x1.clone().detach().requires_grad_(True)
y2 = y1.clone().detach().requires_grad_(True)
z2 = dot(x2, y2)
z2.backward()

diff = z1 - z2
assert (-1e-5 < diff).all()
assert (diff  < 1e-5).all()

assert (x1.grad - x2.grad == 0).all()
assert (y1.grad - y2.grad == 0).all()
