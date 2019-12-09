# PyTorch Module

[PyTorch
Module](https://pytorch.org/docs/stable/nn.html?highlight=module#torch.nn.Module)
has mostly the same role as the custom `Module` class we used before (see e.g.
the [solution to
P6](https://github.com/kawu/hhu-dl-materials/blob/master/solutions/u6/module.py)):
it serves to encapsulate the forward calculation of a network component
together with the corresponding parameters.


## Usage

* Use `super(ClassName, self).__init__()` at the beginning of the
  initialization method of **each class** that (directly or not) inherits from
  the PyTorch Module, where `ClassName` is the name of the current class.
* Always add submodules in the initialization method.  Simply assign them to
  the object's attributes (see the [example below](#example_ffn)).
* In case you want to use a [raw
  tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) as a
  module's parameter, wrap it in the
  [Parameter](https://pytorch.org/docs/master/nn.html#torch.nn.Parameter)
  object.  Then you can treat it as a sub-module and assign to an attribute in
  the initialization method.

<!---
Then, you have to additionally
[register](https://pytorch.org/docs/master/nn.html#torch.nn.Module.register_parameter)
it.
(**TODO**: this may be actually not necessary?)
-->


## Example: FFN

A feed-forward network (with ReLU activation) can be defined as follows:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, idim: int, hdim: int, odim: int):
        super(FFN, self).__init__()
        # Below, we create two `nn.Linear` sub-modules and assign them
        # to the attributes `lin1` and `lin2`.  They get automatically
        # registered as the FFN's sub-modules.
        self.lin1 = nn.Linear(idim, hdim)
        self.lin2 = nn.Linear(hdim, odim)

    def forward(self, x):
        # The following line is equivalent to: h = self.lin1.forward(x)
        y = self.lin1(x)
        # Apply ReLU and the second layer
        return self.lin2(F.relu(y))
```

#### Parameters

A Module encapsulates the model parameters, which you can retrieve using the
[parameters](https://pytorch.org/docs/stable/nn.html?highlight=parameters#torch.nn.Module.parameters)
method.
```python
ffn = FFN(10, 5, 3)
ffn.parameters()        # Generator of parameters
list(ffn.parameters())  # To actually see the parameters
```

#### Application

You can apply an FFN to a vector:
```python
x = torch.randn(10)
y = ffn(x)
y.shape         # => torch.Size([3])
```

[nn.Linear](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear) is
batching-enabled, hence FFN is batching-enabled as well.  You can therefore
apply it to a batch of vectors, too:
```python
X = torch.randn(100, 10)
Y = ffn(X)
Y.shape         # => torch.Size([100, 3])
```

This generalizes to higher-order tensors.  The linear layer (and, consequently,
FFN) always gets applied to the last dimension:
```python
X = torch.randn(100, 50, 10)
Y = ffn(X)
Y.shape         # => torch.Size([100, 50, 3])
```
Note how the last dimension in the example above gets reduced from `10` to `3`
(which is the input and the output size of the created `ffn`).

#### Evaluation mode

The forward calculation and the parameters are not the only things that a
PyTorch Module encapsulates.  Each module also keeps track of the current mode
(training vs evaluation) of the *entire model* (i.e., the main module + all the
submodules).

To retrieve the current mode:
```python
ffn.training    # True by default
assert ffn.training == ffn.lin1.training
                # All modules should be in the same mode
```

You can switch the mode using the `train` or `eval` method of the main module:
```python
ffn.eval()      # Set to evaluation mode
assert ffn.training == ffn.lin1.training
                # The mode of `ffn.lin1` should get updated, too
```

**WARNING**: You should never change the mode of the submodule, because this
will not propagate the mode information to other module components!
```python
ffn.lin1.train()    # Set `ffn.lin1` to training mode
ffn.lin1.training   # => True
ffn.training        # => False
```

Why should you care about the evaluation mode?  It's important for
[dropout](dropout.md)!
