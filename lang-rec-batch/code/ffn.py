# from typing import NamedTuple, Callable
import torch

from core import TT
from module import Module


def stack(x: TT, k: int) -> TT:
    """Stack the given vector `x` in `k` rows.

    >>> x = torch.randn(10)
    >>> X = stack(x, 5)
    >>> print(X.size())
    torch.Size([5, 10])
    """
    x = x.view(1, -1)
    X = torch.cat(
        tuple(x for _ in range(k)),
        dim=0)
    return X


class Linear(Module):
    """Linear transormation layer.

    >>> m = Linear(20, 30)
    >>> input = torch.randn(128, 20)
    >>> output = m.forward(input)
    >>> print(output.size())
    torch.Size([128, 30])
    """

    def __init__(self, idim: int, odim: int):
        """Create a linear transformation layer (matrix + bias).

        Args:
            idim: size of the input vector
            odim: size of the output vector
        """
        # TODO: initialization as in PyTorch Linear
        # TODO: explain why idim first, then odim
        self.register("M", torch.randn(idim, odim))
        self.register("b", torch.randn(odim))

    def forward(self, X: TT):
        # TODO: explain the shape of X
        assert X.shape[1] == self.isize()
        # TODO: explain what happens here
        B = stack(self.b, X.shape[0])
        # TODO: explain why X is on the left
        return torch.mm(X, self.M) + B

    def isize(self):
        """Input size"""
        return self.M.shape[0]

    def osize(self):
        """Output size"""
        return self.M.shape[1]


class FFN1(Module):
    """Feed-forward network with one hidden layer."""

    def __init__(self, idim: int, hdim: int, odim: int):
        """Create a feed-forward network.

        Args:
            idim: size of the input vector
            hdim: size of the hidden vector
            odim: size of the output vector
        """
        self.register("L1", Linear(idim, hdim))
        self.register("L2", Linear(hdim, odim))

    # TODO: you are making an exception and using upper-case for
    # higher-order tensors!
    def forward(self, X: TT) -> TT:
        """Transform the input vector using the underlying FFN."""
        # Explicitely check that the dimensions match
        assert X.shape[1] == self.L1.isize()
        # Calculate the hidden vector
        H = torch.sigmoid(self.L1.forward(X))
        return self.L2.forward(H)


class FFN(Module):
    """Feed-forward network as a network module."""

    def __init__(self, idim: int, hdim: int, odim: int):
        """Create a feed-forward network.

        Args:
            idim: size of the input vector
            hdim: size of the hidden vector
            odim: size of the output vector
        """
        # Note that we don't have to set requires_grad to True, because
        # the register method takes care of that for us.
        # Note, however, that the PyTorch Module class follows slightly
        # different conventions (we will probably switch to using it
        # later).
        self.register("M1", torch.randn(hdim, idim))
        self.register("b1", torch.randn(hdim))
        self.register("M2", torch.randn(odim, hdim))
        self.register("b2", torch.randn(odim))

    def forward(self, x: TT) -> TT:
        """Transform the input vector using the underlying FFN."""
        # Explicitely check that the dimensions match
        assert x.shape[0] == self.M1.shape[1]
        # Calculate the hidden vector
        h = torch.mv(self.M1, x) + self.b1
        return torch.mv(self.M2, torch.sigmoid(h)) + self.b2
