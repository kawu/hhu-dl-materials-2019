# Dropout

[Dropout](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout), applied to
a tensor, randomly zeroes some of the tensor's elements.  Applying dropout to
intermediate results of a PyTorch calculation helps to avoid overfitting.

## Example

```python
import torch
import torch.nn as nn

# Create Dropout with zeroing probability set to 0.25
dropout = nn.Dropout(p=0.25)
# Create a sample tensor vector and apply dropout
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
dropout(x)      # => tensor([0.0000, 2.6667, 0.0000, 5.3333, 0.0000])
```

As you can see, dropout not only zeroes the elements, it also rescales the
tensor so that it has, roughly, the same size (sum of elements) as on input.
You can check this using the following code:
```python
# Average function
def avg(xs):
    return sum(xs) / len(xs)
# Sum of elements on input
x.sum()     # => tensor(15.)
# Average sum of elements with dropout
avg([dropout(x).sum() for _ in range(10**5)])
            # => tensor(14.9963)
```

#### Batching

Dropout applies to higher-order tensors without problem:
```python
x = torch.randn((3, 3, 3))
dropout(x).shape    # => torch.Size([3, 3, 3])
```

## Training vs evaluation

Dropout must be only applied in the training mode.  If you forget
about this detail, your final results may be significantly worse!

Recall [the section on the evaluation mode](module.md#evaluation-mode) in the
document about PyTorch modules.  In the evaluation mode, dropout is equivalent
to an identity function (`lambda x: x`):
```python
dropout.eval()
dropout.training            # => False (not in the training mode)
assert (x == dropout(x))    # dropout(x) just returns x
```

Note that, in practice, `dropout` is virtually never the top-level module of
your PyTorch program, so you should not use `dropout.eval()`, but rather
`main_module.eval()`, where `main_module` is the top-level PyTorch module.

#### Using `with`

It is equally important not to forget to restore the original mode at the end
of evaluation.  A rather elegant solution is to use the [`with`
statement](https://docs.python.org/3/reference/compound_stmts.html#with) and a
[context
manager](https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers)
which guarantee that the original mode is restored (in an exception-safe way).

```python
def eval_on(model: nn.Module):
    """Create a context manager to enter the evaluation mode."""

    class EvalMode:

        def __enter__(self):
            self.mode = model.training
            model.train(False)

        def __exit__(self, _tp, _val, _tb):
            model.train(self.mode)

    return EvalMode()
```

You can use `eval_on` very much like
[`torch.no_grad()`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.no_grad):
```python
dropout.train()
dropout.training            # => True
with eval_on(dropout):
    print(dropout.training) # => False
dropout.training            # => True
```

Again, in practice, apply `eval_on` to the top-level module, so that the
training/evaluation mode information get propagated to all sub-modules.

## Larger example

Let's say we want to extend a [FFN](module.md#example_ffn) with dropout,
applied to the hidden layer.  This can be done as follows:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, idim: int, hdim: int, odim: int, dropout: float = 0.0):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(idim, hdim)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hdim, odim)

    def forward(self, x):
        # Apply the first layer, apply dropout (alternatively, we could
        # apply Dropout after ReLU)
        y = self.dropout(self.lin1(x))
        # Apply ReLU and the second layer (after applying dropout)
        return self.lin2(F.relu(y))
```

Note that, in the sample code above, the `Dropout` object is created in the
initialization function.  This is required because `Dropout` is a PyTorch
module and needs to have access to the evaluation mode information, in
particular.
```python
# Creat an random FFN
ffn = FFN(100, 50, 10, dropout=0.5)
# Create a random input tensor
x = torch.randn(100)
# Make sure we are in the training mode
ffn.train()
# We can be pretty sure that the result will be different in different FFN
# applications because of dropout (pointless exercise: what is the probability
# that the assertion below fails?)
assert (ffn(x) != ffn(x)).all()
# However, if we turn the evaluation mode on, the results are guaranteed to
# be the same, becuase dropout does not apply in this case
ffn.eval()
assert (ffn(x) == ffn(x)).all()
```
