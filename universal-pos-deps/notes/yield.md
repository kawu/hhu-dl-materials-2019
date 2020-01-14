# Intro

This document explains why the following code fragment (from `tagger.py`)
didn't work during the session -- more precisely, why `self.train()` in the
last line was never executed.
```python
    def tags(self, batch: Sequence[Sequence[Word]]) \
            -> Iterable[List[Tuple[POS, Head]]]:
        """Predict the POS tags and dependency heads in the given batch."""
        # Turn evaluation mode on
        self.eval()
        # TODO: does it make sense to use `tag` as part of training?
        with torch.no_grad():
            # POS tagging is to be carried out based on the resulting scores
            pos_scores_batch, dep_scores_batch = zip(*self.forwards(batch))
            # print("train mode is on:", self.training)
            for pos_scores, dep_scores, sent in zip(
                    pos_scores_batch, dep_scores_batch, batch):
                # Predict POS tags and dependency heads
                pos_preds = self.predict_pos_tags(pos_scores)
                dep_preds = self.predict_heads(dep_scores)
                # We should have as many predicted POS tags and dependency
                # heads as input words
                assert len(sent) == len(pos_preds) == len(dep_preds)
                # Return the predicted POS tags
                yield list(zip(pos_preds, dep_preds))
        # Turn evaluation mode off
        self.train()
```

## Yield

Let's define a simple generator function which generates numbers from `0` to
`n-1` and then prints "last line".
```python
def yield_test(n):
    k = 0
    while k < n:
        yield k
        k += 1
    print("last line")
```

If we use it with a for loop, everything works as expected and "last line" is
printed at the end.
```python
for x in yield_test(5):
    print("x =", x)
# x = 0
# x = 1
# x = 2
# x = 3
# x = 4
# last line
```

If we first transform it to a list and then print its length, everything works
fine, too.
```python
xs = list(yield_test(5))
print(len(xs))
# last line
# 5
```


However, if we quit the for loop (e.g., using "break"), even after processing
all the elements, the code after the last `yield` in `yield_test` won't be
executed (and, in particular, "last line" won't be printed).
```python
i = 0
for x in yield_test(5):
    print("x =", x)
    i += 1
    if i >= 5:
        break
# x = 0
# x = 1
# x = 2
# x = 3
# x = 4
```

Similar behavior occurs with `zip`: "last line" is not printed
(which [explains why our code didn't work](https://github.com/kawu/hhu-dl-materials/blob/bf83a277a522ab6100f4feaf8aafdac789dfa1ff/universal-pos-deps/tagger.py#L281-L283)).
```python
xs = range(5)
ys = yield_test(5)
print(list(zip(xs, ys)))
# [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
```


## Context manager

A context manager, as described [on github](https://github.com/kawu/hhu-dl-materials/blob/master/high-api/dropout.md#using-with), seems like a good solution to
the problem -- it guarantees that the model is set to the original mode
(evaluation or training) at the end of the `with` block.
```python
import torch.nn as nn

def eval_on(model: nn.Module):
    """Create a context manager to enter the evaluation mode."""

    class EvalMode:

        def __enter__(self):
            self.mode = model.training
            model.train(False)

        def __exit__(self, _tp, _val, _tb):
            model.train(self.mode)

    return EvalMode()
```

When combined with `yield`, it's still not perfect, though.  The code below
will correctly print "before: True", "inside: False", and "after: True".
However, it will also print "between: False", which shows that the training
mode can be set to `False` also outside of the syntactic `with eval_on(model)`
block.
```python
def test_eval_on(model):
    with eval_on(model):
        print("inside:", model.training)
        yield model

model = nn.Linear(10, 10)

print("before:", model.training)
for x in zip(range(1), test_eval_on(model)):
    print("between:", model.training)
print("after:", model.training)
```

This can lead to unexpected behavior/bugs, so it is probably best to avoid
combining `yield` with the context manager`eval_on`.  Note that this most
likely applies to `torch.no_grad`, too!

<!---
This suggests that `with torch.no_grad` used
[in our code](https://github.com/kawu/hhu-dl-materials/blob/bf83a277a522ab6100f4feaf8aafdac789dfa1ff/universal-pos-deps/tagger.py#L204) is not perfectly safe either, since `yield` is used within the
corresponding syntactic block.  Theoretically, `yield` could cause a kind of a
leakage: the gradient-calculating machinery could be turned off outside of the
`with` block.  This suggests that we should rewrite the tagging method to move
`torch.no_grad` one level lower.
-->
