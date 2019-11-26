# Homework

https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/session6/u6_eng.pdf


# Exercise 1

**Task**: implement a batching-enabled version of a linear layer.

**Solution**: see the implementation of the [Linear](ffn.py#L80-L163) class in
`ffn.py`.  You can also have a look at the
[diff](../../../../commit/353297ff43a7826bb0f5d7710f88aa7e0e5a0520#diff-260db1f91a179af71039b4cd0c33aee2).

**Additional exlanations**: provided in a PDF on the course's website.


# Exercise 2

**(a)**: Have a look at the [forward](main.py#L116-L124) method of the
`LangRec` class, which contains the essential modifications, or at the corresponding
[diff](../../../../commit/353297ff43a7826bb0f5d7710f88aa7e0e5a0520#diff-27b1afc4ec2e27f1b130d69e5c421b28).

**(b)**: Making the EmbeddingSum class work over batches of feature groups
would not necessarily improve the performance of the current solution.  In the
EmbeddingSum class, features (n-grams) are [translated to feature indices
sequentially](embedding.py#L96-L100), before they are [fed to the
EmbeddingBag](embedding.py#L102-L103) provided by PyTorch.

To make the EmbeddingSum work over batches effectively, the training dataset
would have to be pre-processed so as to already contain the [feature
indices](embedding.py#L102).  However, it is not clear how much speed-up this
modification would bring.  It is likely that most of the computation time is
spent in the [forward/backward pass](main.py#L123) of the FFN module which
calculates the scores, in which case further optimization of the EmbeddingSum
wouldn't make a significant difference.
