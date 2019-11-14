from typing import Iterator

import torch

from core import TT


def avg(xs):
    """Calculate the average of the given list."""
    assert len(xs) > 0
    return sum(xs) / len(xs)


def from_rows(xs: Iterator[TT]) -> TT:
    """Stack the given vectors `xs` as rows in a single tensor.

    >>> x = torch.randn(10)
    >>> X = from_rows([x, x, x])
    >>> print(X.size())
    torch.Size([3, 10])
    >>> all(x == X[0])
    True
    """
    xs = tuple(x.view(1, -1) for x in xs)
    return torch.cat(xs, dim=0)


def stack(x: TT, k: int) -> TT:
    """Stack the given vector `x` in `k` rows.

    >>> x = torch.randn(10)
    >>> X = stack(x, 5)
    >>> print(X.size())
    torch.Size([5, 10])
    """
    return from_rows(tuple(x for _ in range(k)))
