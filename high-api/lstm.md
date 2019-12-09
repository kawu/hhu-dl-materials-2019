# LSTM

PyTorch
[LSTM](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM) is
a variant of a recurrent network.  It applies to a sequences of vectors (of
size `n`) and outputs a sequence of modified vectors (of size `m`).

## Example

Let's assume that the input embedding vectors have the size 5 and that we want
to transform them with an LSTM to vectors of size 3.  The first step is to
create an LSTM module:
```python
import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=5, hidden_size=3)
```

Let's create a sequence (matrix) of 10 random input vectors which represent the
embedding vectors of the subsequent words in the input sentence.  Each vector
must have the size of 5 (the input size of the LSTM).
```python
xs = torch.randn((10, 5))
```

If you try to apply the LSTM to the input vector sequence now, you will get an
exception:
```python
lstm(xs)    # => RuntimeError: input must have 3 dimensions, got 2
```
This is because PyTorch LSTM requires that you submit the input sequences in
batches.  The third required dimension is thus the size of the batch.
If we insist that we just want to apply the LSTM to the single `xs` sequence,
we can transform it to a single-element batch.

If you look at the `input` section of the [LSTM
documentation](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM),
you will see that input is supposed to have the shape `(seq_len, batch,
input_size)`.  The batch dimension is thus the second one.  We can transform
`xs` to this representation using
[view](https://pytorch.org/docs/stable/tensors.html?highlight=view#torch.Tensor.view):
```python
xs.view(10, 1, 5).shape     # => torch.Size([10, 1, 5])
```
You can also provide `-1` for one of the dimensions, to make PyTorch infer it
automatically:
```python
xs.view(10, 1, -1).shape     # => torch.Size([10, 1, 5])
```

Now we can apply LSTM to our single-element batch:
```python
xs_batched = xs.view(10, 1, -1)
hs, _ = lstm(xs_batched)
hs.shape        # => torch.Size([10, 1, 3])
```
As you can see, the last dimension -- which represents the actual vectors that
get transformed -- is reduced from 5 (the input size) to 3 (the output/hidden
size).

If we had another vector sequence `ys`, with the same shape as `xs`, we could
apply the LSTM to both of them as follows:
```python
ys = torch.randn((10, 5))
batch = torch.stack([xs, ys])
batch.shape     # => torch.Size([2, 10, 5])
batch = batch.view(10, -1, 5)
batch.shape     # => torch.Size([10, 2, 5])
hs, _ = lstm(batch)
hs.shape        # => torch.Size([10, 2, 3)]
```

#### Dynamic sequence length

TODO

#### Parameters

TODO:
* Layers
* BiLSTMs
* Dropout
* ...
