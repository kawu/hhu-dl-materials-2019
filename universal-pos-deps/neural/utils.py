from typing import Iterator, List, Any

import torch
import torch.nn.utils.rnn as rnn

from neural.types import TT


def avg(xs):
    """Calculate the average of the given list."""
    assert len(xs) > 0
    return sum(xs) / len(xs)


def flatten(xss: List[List[Any]]) -> List[Any]:
    """Flatten the given 2d list.

    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    """
    ys = []
    for xs in xss:
        ys.extend(xs)
    return ys


def round_tt(x: TT, n_digits: int) -> TT:
    """Round the given tensor to `n_digits` decimal places.

    >>> x = torch.tensor([1.11111111])
    >>> all(round_tt(x, 5) == torch.tensor([1.11111]))
    True

    WARNING: this function does not support backpropagation!
    """
    # The function doesn't support backpropagation because
    # `torch.round` itself does not support it.
    return torch.round(x * 10**n_digits) / (10**n_digits)


def unpack_sequence(pseq: rnn.PackedSequence) -> Iterator[TT]:
    """Inverse of rnn.pack_sequence.

    >>> a = torch.tensor([1,2,3])
    >>> b = torch.tensor([4,5])
    >>> c = torch.tensor([6])
    >>> pseq = rnn.pack_sequence([a, b, c])
    >>> pseq #doctest: +ELLIPSIS
    PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), ...)
    >>> useq = list(unpack_sequence(pseq))
    >>> assert (useq[0] == a).all()
    >>> assert (useq[1] == b).all()
    >>> assert (useq[2] == c).all()

    >>> a = torch.tensor([[7,8],[9,10]])
    >>> b = torch.tensor([[1,2],[3,4],[5,6]])
    >>> pseq = rnn.pack_sequence([a, b], enforce_sorted=False)
    >>> useq = list(unpack_sequence(pseq))
    >>> assert (useq[0] == a).all()
    >>> assert (useq[1] == b).all()
    """
    batches, lengths = rnn.pad_packed_sequence(pseq, batch_first=True)
    for batch, length in zip(batches, lengths):
        yield batch[:length]
