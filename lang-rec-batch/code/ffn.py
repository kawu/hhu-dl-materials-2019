import torch

from core import TT
from module import Module
import utils


class SimpleFFN(Module):

    """Feed-forward network with one hidden layer (previous version)."""

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
        # Pay attention to the order of dimensions (idim, odim).  This is
        # related to how we multiply the matrix M by the input vector
        # (see the forward method).
        self.register("M", torch.randn(idim, odim))
        # We use `torch.randn` for simplicity, the Linear class in PyTorch
        # uses a different initialization method (see the docs).
        self.register("b", torch.randn(odim))

    def forward(self, X: TT):
        """Linearly transform the row vectors in X.

        Arguments:
            X: a batch (matrix) of input vectors, one vector per row.
        Returns:
            A batch of output vectors (one vector per fow).
        """
        # Check if the sizes of the row vectors in X match
        # the transformation layer input size.
        assert X.shape[1] == self.isize()
        # Below, each row in B corresponds to the bias vector b.
        B = utils.stack(self.b, X.shape[0])
        # Note that `self.M` is the second argument of `torch.mm`.
        # This is because the input vectors are stored in the rows
        # of the X matrix.
        return torch.mm(X, self.M) + B

    def isize(self):
        """Input size"""
        return self.M.shape[0]

    def osize(self):
        """Output size"""
        return self.M.shape[1]


class FFN(Module):
    """Feed-forward network with one hidden layer.

    A variant of `SimpleFFN` which transforms vectors in batches.
    """

    def __init__(self, idim: int, hdim: int, odim: int):
        """Create a feed-forward network.

        Args:
            idim: size of the input vector
            hdim: size of the hidden vector
            odim: size of the output vector
        """
        self.register("L1", Linear(idim, hdim))
        self.register("L2", Linear(hdim, odim))

    def forward(self, X: TT) -> TT:
        """Transform the batch of input vectors (one vector per row) with
        the underlying FFN.

        Arguments:
            X: batch of input vectors, one vector per row

        Returns:
            A batch of output vectors, one vector per fow.
        """
        # Explicitely check that the dimensions match
        assert X.shape[1] == self.L1.isize()
        # Calculate the batch of hidden vector, one vector per row.  Note
        # that sigmoid is element-wise, thus it will apply to every single
        # element of the result of the first linear transformation
        # (self.L1.forward(X)).
        H = torch.sigmoid(self.L1.forward(X))
        return self.L2.forward(H)
