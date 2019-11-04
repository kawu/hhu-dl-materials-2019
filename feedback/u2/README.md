# Homework

https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session2/u2_eng.pdf, exercise 2


# Common problems

### dtype's do not match

If you get the following (or similar) exception:
```python
RuntimeError: expected device cpu and dtype Float but got device cpu and dtype Long
```
Then you are probably trying to combine tensors with different `dtype`'s.

In PyTorch, certain operations (e.g., `+` or `*`) won't work unless the
arguments store numbers of the same type.  So, for instance, these won't work:
```python
torch.tensor([0, 1]) + torch.tensor([0.0, 1.0])
torch.tensor([0, 1], dtype=torch.float) + torch.tensor([0, 1], dtype=torch.int64)
```

### Backpropagation doesn't work

Even though the matrix-vector product does work.

You should take care of the following:
* The calculation should be based exclusively on (float) tensors.  Don't use regular floats or ints (one exception is when you want to access a particular element of a tensor, e.g. 'x[i]', then you need an int 'i').
* Do not use the .item() method.  This casts a one element tensor to a regular value (float, int).
* Do not combine existing tensors using torch.tensor.  The torch.tensor method should be used only to create new tensors.

Note also that only tensors of floating point dtype can require gradients.
This will end with an exception:
```
x = torch.tensor([0, 0], requires_grad=True)python
```

<!---
### Arguments why backpropagation does not work

In some solutions arguments were raised explaining why backpropagation does not
work in case of a custom matrix-vector product.  I mention them here because
they present some incorrect misconceptions.

##### Gradient descent algorithm

```
In order to make backpropagation work, gradient descent algorithm is required.
```

Gradient descent and backpropagation are two different concepts.  You can
manually calculate the gradients (no backpropagation) and perform gradient
descent.  You can also calculate the gradients using backpropagation, but make
no effort to find the minimum/maximum of the objective function (no gradient
descent).
--->
