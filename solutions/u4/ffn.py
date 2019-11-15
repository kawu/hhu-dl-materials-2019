# from typing import NamedTuple, Callable
import torch

from core import TT
from module import Module


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
