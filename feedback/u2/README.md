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
arguments store numbers of the same type.  So, for instance, these two won't
work:
```python
torch.tensor([0, 1]) + torch.tensor([0.0, 1.0])
torch.tensor([0, 1], dtype=torch.float) + torch.tensor([0, 1], dtype=torch.int64)
```

### Tensors with ints

Only tensors of floating point `dtype` can require gradients.  This will end
with an exception:
```python
x = torch.tensor([0, 0], requires_grad=True)
```

### Backpropagation doesn't work

Even though the matrix-vector product does work.

You should take special care of the following:
<!---
* The calculation should be based exclusively on (float) tensors.  Avoid using regular floats or ints (one exception is when you want to access a particular element of a tensor, e.g. 'x[i]', then you need an int 'i').
-->
* Do not use the `.item()` method.  This casts a one element tensor to a regular value (float, int).
* Do not combine existing tensors using `torch.tensor`.  The `torch.tensor` method should be used only to create new tensors.

For instance, PyTorch won't complain if you do the following to create a tensor
vector with two floats:
```python
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(2., requires_grad=True)
z = torch.tensor([x, y])
```
However, while both `x` and `y` require grad, the `z` tensor does not allow to
perform backpropagation (see the documentation of `torch.tensor` for
explanations).
```python
torch.sum(z).backward() 
# => RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
It's still possible to create `z` in a backpropagable way, using a work-around
with
[view](https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view)
and [cat](https://pytorch.org/docs/stable/torch.html?highlight=cat#torch.cat)
(concatenation):
```python
x           # => tensor(2., requires_grad=True)
x.view(1)   # => tensor([2.], grad_fn=<ViewBackward>)
z = torch.cat([x.view(1), y.view(1)])
torch.sum(z).backward()
```

<!---
To give an example, let's say you have two vectors `v` and `w` of the same size
and you want to create a matrix where `v` is the first row and `w` is the
second row:
```python
v = torch.tensor([0, 1, 2], requrires_grad=True)
w = torch.tensor([2, 1, 0], requrires_grad=True)
torch.tensor([v, w]) # =>
```
However, it uses a regular float (`res`) to store the result and, therefore,
does not allow for backpropagation.
```python
v = torch.randn(5, requires_grad=True)
vsum(v)   # => 
```
-->

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
