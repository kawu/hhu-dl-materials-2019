# Tensors

This document covers the basic tensor operations you should get familar with
today.


## Tensor Object

Tensors can be understood as multi-dimensional arrays containing values of a
certain type (integer, float).
* 0-order tensor is a single value 
* 1-order tensor is a vector of values (one dimensional array)
* 2-order tensor is a matrix of values (two dimentional array)
and so on.

For instance: 
```python
# First import the torch module from the PyTorch framework
import torch

# Create a tensor which contains a single value `13`
value_13 = torch.tensor(13)
value_13              # => tensor(13)

# Create a vector with the values in range(1, 10)
vector_1_to_10 = torch.tensor(range(1, 10))
vector_1_to_10        # => tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create a 3x3 matrix with the same values
matrix_1_to_10 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_1_to_10        # => tensor([[1, 2, 3],
                      #            [4, 5, 6],
                      #            [7, 8, 9]])
```

You can also create tensors of higher dimensions (e.g. ,,cubes'' of values):
```python
```
But we will not need this functionality today.

### Shape

Each tensor has a *shape*, which describes its dimensions.  You can use the
`shape` attribute to access it.
```python
value_13.shape          # => torch.Size([])
value_13.shape          # => torch.Size([])
vector_1_to_10.shape    # => torch.Size([9])
matrix_1_to_10.shape    # => torch.Size([3, 3])
```

You can treat the `torch.Size` objects as regular lists:
```python
len(value_13.shape)           # => 0
list(matrix_1_to_10.shape)    # => [3, 3]
matrix_1_to_10.shape[0]       # => 3
```

You cannot create tensors with irregular shapes.  This is not allowed:
```python
irregular_tensor = torch.tensor(
    [[1, 2, 3], 
     [4, 5, 6],
     [7, 8, 9, 10]])
# => ValueError: expected sequence of length 3 at dim 1 (got 4)
```


## Access

To access elements of tensors, you can basically treat them as lists (of lists
(of lists)).
```python
# Extract the first element of our vector
vector_1_to_10[0]         # => tensor(1)

# Print all the elements in the vector
for x in vector_1_to_10:
    print(x)
# tensor(1)
# tensor(2)
# ...
# tensor(9)

# The slicing syntax also works
vector_1_to_10[:3]        # => tensor([1, 2, 3])

# You can do the same with the matrix (then think of it as a list of lists)
for row in matrix_1_to_10: 
    print(row) 
# tensor([1, 2, 3])
# tensor([4, 5, 6])
# tensor([7, 8, 9])

# TODO: are these rows or columns!?

# Extract the 3rd element of the 3rd row
x = matrix_1_to_10[2][2]
x                         # => tensor(9)
```

Whenever you access some parts or elements of tensors, you still get tensors in
return.  This is important because PyTorch models take the form of computations
over tensors, and the result of these these computations must typically be a
tensor, too.  The fact that, say, `vector_1_to_10[:3]` is a tensor means that
you can easily use it as a part of the computation of the PyTorch model.

You can extract the values from one element tensors if you want, though.
```python
# You can extract the value of a one element tensor using `item()`
x.item()                  # => 9, regular int
# It doesn't work for higher-dimentional tensors
matrix_1_to_10.item()     
# => ValueError: only one element tensors can be converted to Python scalars
```

### dtype and device

So far, we were creating tensors of integers.  You can also create tensors of
floats.
```python
# It's enough that one of the values is a floating-point number
float_vect = torch.tensor([1, 2, 2.5])
float_vect                # => tensor([1.0000, 2.0000, 2.5000])

# You can use the `dtype` attribute to enforce that the values be integers
int_vect = torch.tensor([1, 2, 2.5], dtype=torch.int64)
int_vect                  # => tensor([1, 2, 2])

# ... or floats
ints_as_floats_vect = torch.tensor([1, 2, 3], dtype=torch.float)
ints_as_floats_vect       # => tensor([1., 2., 3.])
```

The target device (CPU, GPU) can be specified for each tensor separetely using
the `device` attribute.
```pyhon
# To run on CPU 
torch.tensor(0, device=torch.device("cpu"))

# To run on GPU (this will probably throw an exception on lab computers,
# where PyTorch was probably not compiled with CUDA enabled)
torch.tensor(0, device=torch.device("cuda:0"))
```
We will be using CPUs throughout the course.  PyTorch defaults to CPUs, so you
don't really have to specify this explicitly each time you create a tensor (it
won't hurt, though).


### requires\_grad


### Randomness


### View

*NOTE*: This part concerns internal representation of tensors.  This will be
quite useful during future sessions, but you can skip it on first reading.

TODO



## Tensor Operations

### Backpropagation
