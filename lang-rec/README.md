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
  name.  The result is a vector of scores (real numbers), with one entry per
  language (English, German, ...).

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
pairs, which corresonds rather directly to its representation on the disk.

*Note*: the name suggests that a *dataset* is a *set* while, in reality, it is
a *sequence* (it can contain duplicates).

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
without any additional boilerplate code.   This is useful when you train a
neural network, where you need to determine the set of network's parameters in
order to update them w.r.t. their gradients (see the `train` function in
`train.py`).

For instance, to create a linear transformation layer with a bias vector:
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

Create a `Layer` which transforms vectors of size 3 to vectors of size 2:
```python
layer = Layer(3, 2)

# The parameters are registerd as `M` and `b` attributes.  Note that the
# registered tensors have `requires_grad` automatically set to True for
# convenience (it's easy to forget this).
layer.M
# => tensor([[-1.8877, -0.9792,  0.6295],
#            [ 0.0668,  0.9066, -1.4973]], requires\_grad=True)
layer.b
# => tensor([ 0.6011, -1.3698], requires\_grad=True)
```

You can retrieve the set of all parameters using the `params()` method (note
that the resulting list contains `layer.M` and `layer.b`):
```python
layer.params()
# =>
#    [tensor([[-1.8877, -0.9792,  0.6295],
#             [ 0.0668,  0.9066, -1.4973]], requires\_grad=True),
#     tensor([ 0.6011, -1.3698], requires\_grad=True)]
```

Transform a 3-element vector to a 2-element vector:
```python
x = torch.tensor([1., 2., 1.])
y = layer.forward(x)
y                     # => tensor([-2.6154, -0.9871], grad_fn=<AddBackward0>)
```
Note that the `grad_fn` attribute is set, which means that we can use the
tensor `y` in subsequent calculations and still be able to calculate the
gradient for the layer's parameters `layer.M` and `layer.b`.

**Modules in PyTorch**.  The network modules in PyTorch are implemented in a
similar manner in that they encapsulate the forward calculation with the
corresponding parameters.  In PyTorch, however, it's not necessary to
explicitely `register` the parameters (the `requires_grad=True` should be used
instead).  We will use the simpler version with explicit registration for now
and switch to PyTorch modules later.

### ffn.py

This Python module implements a feed-forward network (FFN) wrapped as a network
`Module`.

For instance, to create a FFN which transforms 3-element vectors to 2-element
vectors, with a single 5-element hidden layer:
```python
ffn = FFN(idim=3, odim=2, hdim=5)
```

If you look at the implementation, you will see that four tensor are registered
as FFN's parameters: `M1`, `b1`, `M2`, and `b2`.  You can access them as
attributes, for instance:
```python
ffn.b1    # => tensor([-1.0956, -0.8147,  1.8708, -1.0773, -0.7101], requires_grad=True)
```
You can also use `ffn.params()` to retrieve the list of the `ffn`'s tensor
parameters.

**Exercise**.  It is possible to implement FFN as a combination of two
`Layer`'s (see above), with additional application of a non-linear (e.g.,
sigmoid) function in-between.  You can try doing that and see that the result
is pretty much the same for all the practical purposes.

### embedding.py

This Python module provides en embedding dictionary, which allows to map
symbols in a given, pre-computed alphabet (e.g., the set of characters that
occur in person names) to the corresponding vectors.

For instance:
```python
# Let's say we want to embed three characters: 'a', 'b', and 'c'
symset = set(['a', 'b', 'c'])

# Each character is to be mapped to a 3-element vector
emb = Embedding(symset, emb_size=3)

# Let's see the vector assigned to 'a'
emb.forward('a')  # => tensor([-0.6354, -0.2746, -0.6508])
```

During testing, it may be that we will try to embed a symbol we didn't see
during training.  Since characters have no internal structure (at least for
European languages), it makes sense to embed out-of-vocabulary (OOV) chars as
zero vectors.  Given that we want to use CBOW (sum of vectors) representation
for names, zero vectors will have no impact on the resulting representations.
```python
# Embedding for a character outside of the pre-determined alphabet
emb.forward('x')  # => tensor([0., 0., 0.])
```

### encoding.py

The objective of the `Encoding` class is to create a (bijective) mapping
between the languages (English, German, etc.) and the corresponding, unique
integer values from `{0, 1, ..., m-1}`, where `m` is the number of languages.

For instance:
```python
classes = ["English", "German", "French"]
enc = Encoding(classes)
enc.encode("English")   # => 0
```

This mapping is one-to-one and the identifiers cover the range `{0, 1, 2}`:
```python
assert "English" == enc.decode(enc.encode("English"))
assert "German" == enc.decode(enc.encode("German"))
assert "French" == enc.decode(enc.encode("French"))
assert set(range(3)) == set(enc.encode(cl) for cl in classes)
```

The need for such a mapping is motivated as follows.  We want to map any given
name (sequence of characters) to a score vector with `m` elements.  Each entry
in the resulting vector is to represent the score for a particular language
(the higher the score, the more likely the name is to belong to the
corresonding language).  We therefore need to relate the names (strings) with
positions in the score vector, so that we know that, for instance, the score at
the first position corresponds to English, the score at the second position
corresponds to German, and so on.

### main.py

The `main.py` module currently contains:   
* The language recognition network module (`LangRec`), currently a stub
* Functions for calculating the loss of the network w.r.t. a training dataset
  (`single_loss` and `total_loss`)
* The training function `train`, which uses gradient descent to (try to) find
  the parameters of the network module `LangRec` which minimze the `total_loss`
  over the given training dataset.
* Helper functions to print the predictions of the model (`print_predictions`)
  and calculate the accuracy of the model on the given dataset (`accuracy`)
* The `main` function, with the instructions of the script

##### LangRec

`LangRec` is a network `Module`, hence we need to:
* Implement `__init__`, where we define and `register` the parameters of the
  language recognition network.
* Implement `forward`, where we define the steps to calculate the score vector
  (with one real number per language) for the given person name (string).
  Remember that this function should be based on tensors (so that it is
  possible to backpropagate from the resulting score vector to the parameters
  of the network).

`LangRec` also contains (stubs of) two auxiliary methods:
* `encode`: to map a language (string) to its corresponding, unique int
* `classify`: to determine the (language -> probability) mapping for a given
  name
We don't need them to train the network, but they will be useful to preform
predictions once we determine the network's parameters.

##### Loss

There are two functions to calculate the *loss*.  The `single_loss` function
calculates the loss for a single (name, language) pair.  As arguments, it takes
the `output` vector with scores (one score per language), and the `target`
language we would like our network to predict.  The `target` language is
represented by the unique integer (the corresponding position in the score
vector), obtained via [`Encoding`](#encoding).

Formally, the loss for a single (name, language) pair is defined as the [cross
entropy loss](https://en.wikipedia.org/wiki/Cross_entropy) between the predicted probability distribution and the
target probability distribution.  Internally, `single_loss` uses
[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss),
which takes the predicted score vector instead of the predicted distribution
and the target index instead of the target distribution.  Additionally, the
its arguments are higher-order tensors (hence the use of `view`), which allows
for batching.  Don't worry about the details for now, we will get back to
batching, `view`, and cross-entropy loss later.

##### Training

Training is performed using gradient descent.  Iteratively:
* Calculate the loss over the entire dataset
* Calculate the gradients w.r.t. the loss
* Move in the opposite direction of the gradients
The third step is faciliated by wrapping the network in the `Module` class,
which makes it easy to retrieve the set of parameters using the `params()`
method.


[design]: https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session3/u3_eng.pdf
