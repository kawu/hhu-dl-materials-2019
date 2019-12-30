# Optimization

This document describes the steps implemented in order to optimize the POS
tagger.


### Baseline

Baseline is the tagger [implemented before the Christmas
break](https://github.com/kawu/hhu-dl-materials/tree/e1f252990fb01cd5e3a36d2b20b9f932aaccc625).
You can measure its performance in IPython:
```
In [1]: timeit -n 1 -r 3 run main
...
4min 38s ± 1.9 s per loop (mean ± std. dev. of 3 runs, 1 loop each)
```
Below, we compare the speed of the gradually optimized implementation with the
baseline.

<!---
You can get different numbers in absolute terms, of course, depending on the
machine you run the experiments on.
-->

Note that the optimizations described below do not change the model.  All the
hyperparameters (the size of the hidden layer, the size of the embeddings,
etc.) are also unchanged.  Therefore, the resulting accuracy should be also the
same (modulo randomization, which makes the results differ even for the same
implementation of the model).


### Embeddings

In the baseline implementation of the generic embedding class:
* Whenever an out-of-vocabulary (OOV) word is encountered, a [zero embedding vector
  is created explicitly](https://github.com/kawu/hhu-dl-materials/blob/e1f252990fb01cd5e3a36d2b20b9f932aaccc625/universal-pos-deps/neural/embedding.py#L61-L62).
* Vocabulary elements (in our case -- words) are always [embedded individually](https://github.com/kawu/hhu-dl-materials/blob/e1f252990fb01cd5e3a36d2b20b9f932aaccc625/universal-pos-deps/neural/embedding.py#L69-L72).

We can improve on these two points by, respectively:
* Using the padding index of the [PyTorch Embedding
  class](https://pytorch.org/docs/stable/nn.html#embedding).  See the corresponding code [here](https://github.com/kawu/hhu-dl-materials/blob/b5e57f73e0eb6ee58dd049a9e0f07ca0c477507e/universal-pos-deps/neural/embedding.py#L49-L60) and [here](https://github.com/kawu/hhu-dl-materials/blob/b5e57f73e0eb6ee58dd049a9e0f07ca0c477507e/universal-pos-deps/neural/embedding.py#L71-L73).
* [Embedding words in groups](https://github.com/kawu/hhu-dl-materials/blob/b5e57f73e0eb6ee58dd049a9e0f07ca0c477507e/universal-pos-deps/neural/embedding.py#L76-L84).

The former allows to avoid explicitly creating zero embedding vectors.  It also
enables the latter optimization -- embedding words in groups -- which allows to
avoid stacking together (with `torch.stack`) the resulting embedding vectors.
See also [the corresponding diff](https://github.com/kawu/hhu-dl-materials/commit/b5e57f73e0eb6ee58dd049a9e0f07ca0c477507e#diff-61ae524f1b0f2b45b8f89e7ff015956e) for details.

<!---
If we first create the embedding vectors and then stack them together, as in
the baseline implementation, the `backward` method of the `torch.stack`
function has to be used during backpropagation.  When we embed words in groups,
`torch.stack` is not used and, consequently, backpropagation is faster.
-->

<!---
If this is surprising, note that when several words are embedded together, we
first calculate the indices corresponding to the individual words, which does
not involve backpropagation because the indices are fixed, we only adapt the
corresponding embedding vectors during training.  Hence, the backward method of
the Embedding class also works ,,in a batch'', i.e., for the entire group of
words in parallel.
-->

As a result of this optimization:
```
In [1]: timeit -n 1 -r 3 run main
...
3min 29s ± 202 ms per loop (mean ± std. dev. of 3 runs, 1 loop each)
```

<!---
TODO: consider embedding for several sentences at the same time.
-->


### LSTM

As described on the [page about
LSTMs](https://github.com/kawu/hhu-dl-materials/blob/dev/high-api/lstm.md#dynamic-sequence-lengths),
applying LSTM to a batch of sentences is not trivial because the lengths of the
individual sentences can differ.  To this end, a
[PackedSequence](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.utils.rnn.PackedSequence)
can be used.

This optimization involves:
* [Adding the `forwards` method](https://github.com/kawu/hhu-dl-materials/blob/6351c5e1cd4333fa4bfb2f86c59616dd4cd58d64/universal-pos-deps/main.py#L73-L110) to the POS tagger, which processes sentences in
  batches using the packed sequence representation.
  <!--- 
  (of course, you could use a different name for this method)
  -->
* [Using the new `forwards` method in the `total_loss` function](https://github.com/kawu/hhu-dl-materials/blob/6351c5e1cd4333fa4bfb2f86c59616dd4cd58d64/universal-pos-deps/main.py#L178-L181), to actually
  processes sentences in batches during training.

See also [the corresponding diff](https://github.com/kawu/hhu-dl-materials/commit/6351c5e1cd4333fa4bfb2f86c59616dd4cd58d64#diff-39e3f0a6559bc7cfeea0212650b872f4) for details.
As a result of this optimization:
```
In [1]: timeit -n 1 -r 3 run main
...
43.8 s ± 714 ms per loop (mean ± std. dev. of 3 runs, 1 loop each)
```
