from typing import Iterator

import torch

from core import TT


def avg(xs):
    """Calculate the average of the given list."""
    assert len(xs) > 0
    return sum(xs) / len(xs)


def round_tt(x: TT, n_digits: int) -> TT:
    """Round the given tensor to `n_digits` decimal places.

    >>> x = torch.tensor([1.11111111])
    >>> all(round_tt(x, 5) == torch.tensor([1.11111]))
    True

    WARNING: this function does not support backpropagation.
    """
    # The function doesn't support backpropagation because
    # `torch.round` itself does not support it.
    return torch.round(x * 10**n_digits) / (10**n_digits)


def stack(xs: Iterator[TT]) -> TT:
    """Stack the given vectors `xs` as rows in a tensor matrix.

    >>> x = torch.randn(10)
    >>> X = stack([x, x, x])
    >>> print(X.size())
    torch.Size([3, 10])
    >>> all(x == X[0])
    True
    """
    xs = tuple(x.view(1, -1) for x in xs)
    return torch.cat(xs, dim=0)
