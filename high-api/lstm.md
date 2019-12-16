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

#### Dynamic sequence lengths

The issue with the example above is that, typically, the sequences in the input
batch have different lengths.  For instance, these sequences can correspond to
different sentences, with different numbers of words.

A
[PackedSequence](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.utils.rnn.PackedSequence)
allows to pack a batch of sequences in two tensors:
* the vector of input (embedding) vectors, concatenated together
* the vector of batch lengths

To create a packed sequence,
[pack\_sequence](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_sequence)
or
[pack\_padded\_sequence](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence)
can be used.  For instance:
```python
import torch.nn.utils.rnn as rnn

# Create two "seqeunces", the second shorter than the first
xs = torch.randn((10, 5))
ys = torch.randn((8, 5))
# Convert [xs, ys] to a packed sequence
packed_inp = rnn.pack_sequence([xs, ys])
```
You can access the underlying data structures using `data` and `batch_sizes`
attributes:
```python
packed_inp.data.shape         # => torch.Size([18, 5])
packed_inp.batch_sizes.shape  # => torch.Size([10])
```
In particular, `packed_inp.data` contains all the 18 input vectors (each of
size 5) packed into one vector tensor.  The length of `packed_inp.batch_sizes`,
on the other hand, is the length of the longest input sequence (here,
`len(xs)`).

The important point is that a packed sequence allows to apply the LSTM to the
batch of sequences of different lengths.
```python
packed_hid, _ = lstm(packed_inp)
packed_hid.data.shape         # => torch.Size([18, 3])
```
The output (hidden) vectors are also stored in a packed sequence.  It can be
converted back to a padded representation using
[pad\_packed\_sequence](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence):
```python
padded_hid, padded_len = rnn.pad_packed_sequence(packed_hid, batch_first=True)
padded_hid.shape              # => torch.Size([2, 10, 3])
padded_len                    # => tensor([10,  8])
```
where `padded_hid` contains two output hidden vectors of length `10` each
(the shorter is padded with `0` tensors), and `padded_len` contains the actual 
length of the individual sequences.

<!---
For instance:
```python
# The hidden output tensor corresponding the the `ys` input sequence
hid_for_ys = padded_hid[1]
# Since `len(ys) < 10`, the last two elements should be 0s
hid_for_ys[9] == 0           # => tensor([True, True, True])
hid_for_ys[8] == 0           # => tensor([True, True, True])
hid_for_ys[7] == 0           # => tensor([False, False, False])
```
-->

In general, you can convert the padded representation into a regular list of
variable-length tensors using, for example:
```python
hs = []
for hidden, length in zip(padded_hid, padded_len):
    hs.append(hidden[:length])
```
In our example with two inputs, `xs` and `ys`, this will lead to:
```python
len(hs)                     # => 2  (size of the batch)
len(hs[0])                  # => 10 (length of `xs`)
len(hs[1])                  # => 8  (length of `ys`)
```

**Note**: it the list of input vector sequences is not ordered by length, you
have to add `enforce_sorted=False` as an argument of `pack_sequence`:
```python
# Create two "seqeunces", the first shorter than the second
xs = torch.randn((7, 5))
ys = torch.randn((8, 5))
# Convert [xs, ys] to a packed sequence
packed_inp = rnn.pack_sequence([xs, ys], enforce_sorted=False)
```
Otherwise, you will get an exception.

#### Parameters

TODO:
* Layers
* BiLSTMs
* Dropout
* ...
