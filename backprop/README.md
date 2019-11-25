# Back-propagation

This page provides exercises and examples on how to implement custom
back-propagable functions in PyTorch.  Such functions are called
*autograd* functions in PyTorch (which presumably stands for *automatic gradient*
computation, for which backpropagation is used).  You will very rarely need to
manually implement autograd functions (in contrast to, e.g., batching that we
have seen last week).  Nevertheless, there are situations where this is
necessary.  For instance:
* You may want to use a primitive function not implemented in PyTorch yet
  (by *primitive* I mean a function that is not easily expressible as a
  composition of already available functions)
* Automatically derived backward calculation may be not optimal for certain
  combinations of neural functions


### Useful links

The PyTorch documentation page which contains more detailed information about
writing custom autograd functions can be found at:
https://pytorch.org/docs/stable/notes/extending.html

Some code fragments were borrowed from:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


### Preparation

The commands and code fragments shown below are intended to be used
iteractivelly in IPython.  Besides, the entire code for this session (with the
missing place-holders for the exercise solutions) is placed in the
[backprop.py](backprop.py) file.

First, we will use the following preamble:
```python
from typing import Tuple

import torch
from torch.autograd import Function

# Tensor type, as usual
TT = torch.TensorType
```

`Function` is the class we have to inherit from when we want to define custom
autograd functions in PyTorch.

<!---
However, you may want to perform the exercises below iteractivelly in IPython.
-->


# Examples

### Addition

Let's start with a simple example: element-wise addition.  Of course it is
already implemented in PyTorch, which will allow us to test if our
implemenations work as intended.

```python
class Addition(Function):

    @staticmethod
    def forward(ctx, x1: TT, x2: TT) -> TT:
        y = x1 + x2
        return x1 + x2
        
    @staticmethod
    def backward(ctx, dzdy: TT) -> Tuple[TT, TT]:
        return dzdy, dzdy
```
In the `forward` pass, we receive two tensors that we want to add together:
`x1` and `x2`.  To get the result of the forward method, we simply add them and
return the result.

In the `backward` pass we receive a Tensor containing the gradient of the loss
`z` (whatever it is!) w.r.t the addition result `y`.  We call it `dzdy`.  Now,
we need to calculate the gradients for `x1` and `x2` and return them as a
tuple, in the same order as in the `forward` method.  Using the chain rule, we
can determine that this is just `dzdy` for both `x1` and `x2` (take a moment to
verify this!).

The addition function is now available via `Addition.apply`.  For brevity, it
is recommended to use an alias for custom autograd functions.  In this case:
```python
add = Addition.apply
```

We can now check that our custom addition behaves as the one provided in
PyTorch.
```python
x1 = torch.tensor(1.0, requires_grad=True)
y1 = torch.tensor(2.0, requires_grad=True)
(x1 + y1).backward()
```

We do the same with our custom addition function.
```python
x2 = torch.tensor(1.0, requires_grad=True)
y2 = torch.tensor(2.0, requires_grad=True)
add(x2, y2).backward()
```

And we verify that the gradients match.
```python
assert x1.grad == x2.grad
assert y1.grad == y2.grad
```

The nice part is that, since addition is element-wise, this should work also
for complex tensors, and not only for one-element tensors!  Let's see:
```python
x1 = torch.randn(3, 3, requires_grad=True)
y1 = torch.randn(3, 3, requires_grad=True)
(x1 + y1).sum().backward()

x2 = torch.randn(3, 3, requires_grad=True)
y2 = torch.randn(3, 3, requires_grad=True)
add(x2, y2).sum().backward()

assert (x1.grad == x2.grad).all()
assert (y1.grad == y2.grad).all()
```

### Sigmoid

Let's see another example: the sigmoid (logistic) function.

Let `x` be the input tensor, to which we apply (element-wise) the sigmoid
function.  Let `y` be the result of this application.  Let also `z` be the loss
value.

The derivative of sigmoid, `y = sigmoid(x)`, is `dy/dx = y * (1 - y)`.  From
the chain rule we have `dz/dx = dz/dy * dy/dx`.  Since in the backward
computation we already know `dz/dy`, we need to also know `y` (i.e., the result
of the forward computation) to calculate `dz/dx`.

In the `forward` and `backward` methods, `ctx` is a context object that can be
used to stash information for the backward computation.  You can cache
arbitrary objects for use in the backward pass using the
`ctx.save_for_backward` method.  In our case, we can use it to stash `y` for
the subsequent backward computation.  All this leads to:
```python
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
```

To test it:
```python
x1 = torch.randn(3, 3, requires_grad=True)
torch.sigmoid(x1).sum().backward()

x2 = x1.clone().detach().requires_grad_(True)
sigmoid(x2).sum().backward()

# Check if the difference between the two gradients is sufficiently similar
# (clearly the backward method of the PyTorch sigmoid is better in terms
# of numerical precision).
diff = x1.grad - x2.grad
assert (-1e-7 < diff).all()
assert (diff  < 1e-7).all()
```

# Exercises

### Sum

Re-implement `torch.sum` as a custom autograd function.

### Dot product

Re-implement `torch.dot` as a custom autograd function.

### Matrix-vector product

Re-implement `torch.mv` as a custom autograd function.

**WARNING**. This one is a difficult exercises!
