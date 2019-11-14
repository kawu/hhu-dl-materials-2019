import torch

from core import TT
from module import Module


# TODO: There's a bug in this module (the Embedding class does not work
# exactly as specified in the description of the exercise).  In this case,
# this won't probably prevent you from getting good results, but can be an
# issue in other applications.


class Embedding(Module):

    """A simple lookup table that stores embeddings
    of a fixed dictionary and size.

    >>> import torch
    >>> symset = set(['a', 'b', 'c'])
    >>> emb = Embedding(symset, emb_size=10)
    >>> emb.forward('a') #doctest: +ELLIPSIS
    tensor(...)
    >>> emb.forward('a').shape
    torch.Size([10])
    """

    def __init__(self, alphabet: set, emb_size: int):
        """Create a random embedding dictionary.

        Arguments:
        * alphabet: set of symbols to embed (characters, words, POS tags, ...)
        * emb_size: embedding size (each symbol is mapped to a vector
            of size emb_size)
        """
        self.emb = dict()
        self.emb_size = emb_size
        for sym in alphabet:
            # NOTE: `requires_grad=True` was lacking in the previous version
            # which made the embedding parameters silently ignored in
            # backpropagation.  Consequently, these parameters where
            # fixed throughout the entire training process.
            self.emb[sym] = torch.randn(emb_size, requires_grad=True)

    def forward(self, sym) -> TT:
        """Embed the given symbol."""
        try:
            return self.emb[sym]
        except KeyError:
            return torch.zeros(self.emb_size)

    # We implement the params method manually because we don't use
    # the Module register method to register the tensor parameters
    # (the parameters are kept in a dictionary, not as attributes).
    def params(self):
        """The list of parameters of the embedding dictionary."""
        return list(self.emb.values())
