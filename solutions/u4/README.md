# Homework

https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session4/u4_eng.pdf


# Hyper-parameter search 

**Task**: determine adequate values of the architecture hyper-parameters:
* embedding size
* size of the hidden layer

**Method**: select the values so as to minimize the loss/maximize the accuracy on
the dev set.

Ideas that came up:
* Manually check several different configurations
  * Sometimes we don't have the necessary resources (time, computation
    power) to do a more systematic search
  * The choice of the configurations can be tricky -- the results often don't
    follow our intuition
* Greedy search: iteratively change the values of single hyper-parameters as
  long as loss/accuracy on the dev set improves
  * Computationally cheap, but doesn't guarantee to find the optimal parameters
* Grid search: if the hyper-parameter search space is small
* Random search: sampling random configurations (from a pre-defined domain)
* Bayesian optimization

[Reference solution](main.py#L300-L350):
* Domains (coarse-grained): [10, 50, 100] for both hyper-parameters
* Grid search, since the hyper-parameter space is small
* Several runs for each configuration, to get more reliable results

Warnings:
* We can over-fit to the dev set
  * Alternative: cross-validation, but it's even more costly
* You should *not* try fixing the random seeds.  On the contrary, to get
  reliable results, it's best to perform several training runs for each
  configuration.  Some configurations can lead to a higher variance of the
  results.
* Different configurations can require different SGD parameters (i.e., the
  number of epochs).  Be sure to check that the models do not under-fit because
  of that before drawing conclusions from the results.

<!---
Some problems:
* 
-->

### Grid search output

Here's my output of the reference (bigram-based) solution (with
`mini_batch_size` set to `256`), slightly reformatted:
```
Train size: 16059
Dev size: 2008

# emb_size=10, hid_size=10
# AVG: loss(train)=0.809, acc(train)=0.774, loss(dev)=0.955, acc(dev)=0.742

# emb_size=10, hid_size=50
# AVG: loss(train)=0.772, acc(train)=0.782, loss(dev)=1.053, acc(dev)=0.726

# emb_size=10, hid_size=100
# AVG: loss(train)=0.84, acc(train)=0.767, loss(dev)=1.249, acc(dev)=0.689

# emb_size=50, hid_size=10
# AVG: loss(train)=0.778, acc(train)=0.782, loss(dev)=0.993, acc(dev)=0.733

# emb_size=50, hid_size=50
# AVG: loss(train)=0.487, acc(train)=0.863, loss(dev)=1.084, acc(dev)=0.726

# emb_size=50, hid_size=100
# AVG: loss(train)=0.368, acc(train)=0.899, loss(dev)=1.342, acc(dev)=0.7

# emb_size=100, hid_size=10
# AVG: loss(train)=0.799, acc(train)=0.778, loss(dev)=1.003, acc(dev)=0.735

# emb_size=100, hid_size=50
# AVG: loss(train)=0.444, acc(train)=0.874, loss(dev)=1.062, acc(dev)=0.735

# emb_size=100, hid_size=100
# AVG: loss(train)=0.293, acc(train)=0.919, loss(dev)=1.374, acc(dev)=0.71
```

<!---
TODO: Can we observe that it's not possible to perform greed search to find the
optimal configuration?
TODO: mention overfitting
-->


# Bigram-based model

Have a look at the implementation of the [LangRec class](main.py#L36-L142) and
the [ngram function](main.py#L19-L33).

### Bug

The bug in the Embedding class was that the Embedding vectors had [no
`requires_grad=True` specified](embedding.py#L42).  As a result, embedding
vectors were silently ignored in backpropagation.  Consequently, these
parameters were fixed throughout the entire training process.

Testing (assertions, unit tests, etc.) is your main friend in the hopeless
fight against such issues, which can be really nasty and truly hard to spot.
Better PyTorch API would be another way to prevent such problems.

This issue in particular can be identified using an [additional
assertion](main.py#L67-L70) in the LangRec class.  PyTorch optimizer could
enforce this, too, but it doesn't for some reason (perhaps to allow the user to
switch gradient updating off for selected model parameters).
