# Cross entropy loss

[CrossEntropyLoss](CEL) is one of main loss criterions provided in PyTorch.  It
is useful whenever, as a result of the application of your [neural
module](module.md) or function, you get a probability distribution which you
want to make as close as possible to some known, target probability
distribution.

The [CrossEntropyLoss](CEL) class in PyTorch, however, requires specific
representation of the two distributions:
* The predicted distribution must have the form of a tensor of **scores**
  rather than **probabilities**.  It is impliciately assumed that
  [softmax][softmax] is applied to convert the scores to probabilities, but you
  rarely need to actually do that explicitely.
* The target distribution takes the form of a single number: the position in
  the target probability distribution with value `1`.  All the other
  probabilities are assumed to be `0`.

**Note**: the second point is actually an significant restriction, because it
prevents from dealing with uncertainty in training data.


## Example

It's important to note that [CrossEntropyLoss](CEL) works in batches.  It takes
two arguments:
* tensor with predicted scores
* tensor with target classes (represented by integers)
both of which should have the same leading dimension.  The size of this
dimension corresponds to the size of the batch.

Let's assume that there are three classes (which could e.g. correspond to three
POS tags: `NOUN`, `VERB`, and `ADJ`).  Let's create two score vectors:
```python
import torch
s1 = torch.tensor([1.0, 0.0, -1.0])
s2 = torch.tensor([0.0, -1.0, 1.0])
```
Which means that in `s1` the first class (`NOUN`) and in `s2` the third class
(`ADJ`) would be predicted, since they have the highest scores.

You can apply softmax to get proper distributions (but, as mentioned above,
there's usually no need to do that in the code).
```python
from torch.nn.functional import softmax
softmax(s1, dim=0)     # => tensor([0.6652, 0.2447, 0.0900])
softmax(s2, dim=0)     # => tensor([0.2447, 0.0900, 0.6652])
```

Let's assume that the target classes are `NOUN` (index `0`) and `VERB` (index
`2`), correspondingly.  That is, the model would predict the noun correctly,
but not the verb.  We can represent the target classes using a single tensor:
```python
targets = torch.LongTensor([0, 2])
```
At this point, we can calculate the loss:
```python
# Create the cross entropy loss object
loss = CrossEntropyLoss()
# Stack the predicted scores into one batch tensor
scores = torch.stack([s1, s2])
# Calculate the actual loss
loss(scores, targets)   # => tensor(0.4076)
```
It's easy to check that, if the scores correspond to targets, the loss is
lower:
```python
s1 = torch.tensor([5.0, -5.0, -5.0])
s2 = torch.tensor([-5.0, -5.0, 5.0])
loss(torch.stack([s1, s2]), targets)   # => tensor(9.0833e-05)
```

[CEL]: https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss "Cross entropy loss"
[softmax]: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softmax "Softmax function"
