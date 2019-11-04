# Problem

The problem we are dealing with is to classify person names (Downie, Brune,
Dubrov, Fontaine, etc.) according to their languages (English, German, Russian,
French, etc.).

The (for the moment, incomplete) implementation of one possible solution is in
the `code` directory.


# Solution

The proposed solution can be summarized as a character-level,
continuous-bag-of-words (CBOW) model, in which:
* Each character of the input name is mapped to the corresponding embedding
  vector.
* The CBOW representation (simply a sum of vectors) is used to transform a
  sequence of vectors (each for one character in the input name) to a single
  vector.  The resulting vector represents the entire name.
* A feed-forward network is used to score the vector representation of the
  given name.  The result is a vector of scores (real numbers), with one entry
  per language (English, German, ...).

More formal description can be found in the <b>Task</b> secion of the
[corresponding PDF][design].


# Architecture

This section describes the architecture of the proposed solution.  The
implementation is modular, with different modules responsible for the different
aspects of the solution.  You should try to understand the modules one by one.
The module which gathers them all together is `main.py`.

You should be able to run all the code fragments below once you enter an
IPython session in the `code` subdirectory and perform `run main`.

### core.py

The `core.py` module provides type aliases for the basic types used in the rest
of the code.  In particular, the tensor type (`TT = torch.TensorType`), aliases
used for names (`Name = str`) and languages (`Lang = str`), and the type of the
dataset (`DataSet = List[Tuple[Name, Lang]]`).

In particular, the latter means that our dataset is a list of (name, language)
paris, which corresonds rather directly to its representation on the disk.

*Note*: the name suggests that a datset is a set while, in reality, it is a
sequence (it can contain duplicates)!

### names.py

The `names.py` module covers the dataset-related functionality.  In particular,
it provides a single function `load_data`.  Once you do `run main` in IPython
(in the `code` subdirectory`), and provided that you have a dataset file
`split/dev80.csv` (which we use for training), you can test it as follows:
```python
# Load the dataset
train_set = names.load_data("split/dev80.csv")

# Query the datset
train_set[0]    # => ['Awksentiewsky', 'Russian']
len(train_set)  # => 1606
train_set[5]    # => ['Ludlow', 'English']

# Retrieve the set of languages
set(lang for (name, lang) in train_set)
# => {'Arabic',
#     'Chinese',
#     'Czech',
#     ...
```

### module.py

This module implements the `Module` class, which provides an abstract
representation of a parametrized network (called *network module* henceforth).
Whenever you create a network module with its own parameters or (parametrized)
sub-components (which should get adapted during training), you should make it
an instance of this class.

The goal of `Module` is to encapsulate together:
* The forward calculation, i.e, the function that the network module implements
* The underlying parameters, which should get adapted during training

When you create a new network module (e.g., a feed-forwrad network, or just a
linear transformation layer with a bias vector), you should do two things:
* In the initialization function (`__init__`), register all the parameters of
  the network module using the `self.register` method.
* Implement the forward calculation -- the function represented by the network
  module.

Registering the parameters/sub-modules provides one simple but important
feature: when you create a network module, you can retrieve all it's parameters
without any additional boilerplate code.  For instance, to create a linear
transformation layer with a bias vector:
```python
class Layer(Module):
    """

    def __init__(self, idim, odim):
        """Create a linear transformation layer with a bias vector.

        Args:
            idim: size of the input vector
            odim: size of the output vector
        """
        # Register the linear transformation matrix
        self.register("M", torch.randn(odim, idim))
        # Register the bias vector
        self.register("b", torch.randn(odim))

    def forward(self, x: TT) -> TT:
        """Perform the transformation using the registered parameters."""
        return torch.mv(self.M, x) + self.b
```

Creat a layer which transforms vector of size 3 to vectors of size 2:
```python
layer = Layer(3, 2)

# The parameters are registerd as `M` and `b` attributes.  Note that the
# registered tensors have `requires_grad` automatically set to True, for
# convenience (it's easy to forget this).
layer.M
# => tensor([[-1.8877, -0.9792,  0.6295],
#            [ 0.0668,  0.9066, -1.4973]], requires\_grad=True)
layer.b
# => tensor([ 0.6011, -1.3698], requires\_grad=True)
```

# We can retrieve the set of parameters using the `params()` method.
layer.params()
# =>
#    [tensor([[-1.8877, -0.9792,  0.6295],
#             [ 0.0668,  0.9066, -1.4973]], requires\_grad=True),
#     tensor([ 0.6011, -1.3698], requires\_grad=True)]
```

Transform a sample 3-element vector to a 2-element vector:
```python
x = torch.tensor([1., 2., 1.])
layer.forward(x)  # => tensor([-2.6154, -0.9871], grad_fn=<AddBackward0>)
```
Note that the `grad_fn` attribute is set, which means that we can use the
tensor `x` in subsequent calculations and still be able to calculate the
gradient for the layer's parameters `layer.M` and `layer.b`.

##### Module in PyTorch

The network modules in PyTorch are implemented in a similar manner in that they
encapsulate the forward calculation with the corresponding parameters.  In
PyTorch, however, it's not necessary to explicitely `register` the parameters
(the `requires_grad=True` should be used instead).  We will use the simpler
version with explicit registration for now and switch to PyTorch modules later.

### ffn.py

This Python module implements a feed-forward network (FFN) wrapped as a neural
network module.

For instance, to create a FFN which transforms 3-element vectors to 2-element
vectors, with a single 5-element hidden layer:
```python
ffn = FFN(idim=3, odim=2, hdim=5)
```

If you look at the implementation, you will see that four tensor are registered
as FFN's parameters: `M1`, `b1`, `M2`, and `b2`.  For instance:
```python
ffn.b1    # => tensor([-1.0956, -0.8147,  1.8708, -1.0773, -0.7101], requires_grad=True)
```

You can also use `ffn.params()` to retrieve the list of all its tensor
parameters.

### embedding.py

### encoding.py

### main.py



[design]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session3/u3_eng.pdf
