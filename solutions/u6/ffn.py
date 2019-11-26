import torch

from core import TT
from module import Module
import utils
from utils import round_tt  # round_tt is used in doctests


class LinearNoBatching(Module):

    """Linear transormation layer.  No batching.

    Search for the `Linear` class below for a version which
    supports batching.

    Example:
    >>> m = LinearNoBatching(20, 30)
    >>> input = torch.randn(20)
    >>> output = m.forward(input)
    >>> print(output.size())
    torch.Size([30])
    """

    def __init__(self, idim: int, odim: int):
        """Create a linear transformation layer (matrix + bias).

        Args:
            idim: size of the input vector
            odim: size of the output vector
        """
        # Linear transformation matrix
        self.register("M", torch.randn(odim, idim))
        # Bias vector
        self.register("b", torch.randn(odim))

    def forward(self, x: TT):
        """Perform the linear transformation of the input vector."""
        # Explicitely check that the dimensions match
        assert x.shape[0] == self.isize()
        # Perform the linear transformation and add the bias vector
        return torch.mv(self.M, x) + self.b

    def isize(self):
        """Input size"""
        return self.M.shape[1]

    def osize(self):
        """Output size"""
        return self.M.shape[0]


class FFNNoBatching(Module):

    """Feed-forward network with one hidden layer.  No batching.

    Search for the `FFN` class below for a version which
    supports batching.
    """

    def __init__(self, idim: int, hdim: int, odim: int):
        """Create a feed-forward network.

        Args:
            idim: size of the input vector
            hdim: size of the hidden vector
            odim: size of the output vector
        """
        # FFN is a combination of two linear layers
        self.register("L1", LinearNoBatching(idim, hdim))
        self.register("L2", LinearNoBatching(hdim, odim))

    def forward(self, x: TT) -> TT:
        """Transform the input vector using the underlying FFN."""
        # Calculate the hidden vector first
        h = torch.sigmoid(self.L1.forward(x))
        # Then apply the second layer
        return self.L2.forward(h)


# TODO: your job is to re-implement this class.  Currently, it
# performs all the calculations sequentially.
class Linear(Module):

    """Linear transormation layer.  Batching enabled.

    Example:
    >>> m = Linear(20, 30)
    >>> input = torch.randn(128, 20)
    >>> output = m.forward(input)
    >>> print(output.size())
    torch.Size([128, 30])

    If we apply the layer to each of the vectors separately, we should
    get exactly the same result:
    >>> for i in range(128):
    ...     # Use `view` to make a single-element batch
    ...     input_i = input[i].view(1, -1)
    ...     output_i = m.forward(input_i)[0]
    ...     # Round to disregard small floating-point inaccuracies.
    ...     assert all(round_tt(output_i - output[i], 3) == 0)
    """

    # TODO: re-implement this method.
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

    # TODO: re-implement this method.
    def forward(self, X: TT):
        """Linearly transform the row vectors in X.

        Arguments:
            X: a batch (matrix) of input vectors, one vector per row.
        Returns:
            A batch of output vectors (one vector per fow).
        """
        # Explicitely check that the dimensions match
        assert X.shape[1] == self.isize()

        # Solution 4:
        return torch.mm(X, self.M) + self.b

        # # Solution 3: using transposition of the parameter matrix + placing the
        # # input matrix on the left.  That's almost as good as we can do, but
        # # it's still possible to avoid the transposition.  Also, remember that
        # # you still need to account for the bias vector.
        # return torch.mm(X, self.M.t())

        # Solution 2: using transposition of the input matrix
        # # Transpose the input matrix
        # X = X.t()
        # # Perform the linear transformation
        # Y = torch.mm(self.M, X)
        # # Return the output matrix, after transposition
        # return Y.t()

        # Solution 1: sequential (not what we want)
        # # Calculate the output for each input row separately
        # ys = []
        # for x in X:
        #     y = torch.mv(self.M, x) + self.b
        #     ys.append(y)
        # # Stack the outputs together
        # return utils.stack(ys)

    def isize(self):
        """Input size"""
        return self.M.shape[0]

    def osize(self):
        """Output size"""
        return self.M.shape[1]


class FFN(Module):
    """Feed-forward network with one hidden layer.  Batching enabled.

    Example:
    >>> m = FFN(10, 10, 5)
    >>> input = torch.randn(128, 10)
    >>> output = m.forward(input)
    >>> print(output.size())
    torch.Size([128, 5])

    If we apply the network to each of the vectors separately, we should
    get exactly the same result:
    >>> for i in range(128):
    ...     # Use `view` to make a single-element batch
    ...     input_i = input[i].view(1, -1)
    ...     output_i = m.forward(input_i)[0]
    ...     # Round to disregard small floating-point inaccuracies.
    ...     assert all(round_tt(output_i - output[i], 3) == 0)
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
