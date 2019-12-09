from typing import Iterator

import torch
import torch.nn as nn

from core import TT
from module import Module
from encoding import Encoding


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


class EmbeddingSum(Module):

    """A lookup table that stores embeddings of a fixed dictionary and size,
    combined with summing (CBOW).

    EmbeddingSum is an optimized variant of the Embedding class combined
    with summing (CBOW).  It is intended to be used over bags of features
    rather than single features.  It is based on torch.nn.Embedding
    (look it up in PyTorch docs).

    >>> import torch
    >>> symset = set(['a', 'b', 'c'])
    >>> emb = EmbeddingSum(symset, emb_size=10)
    >>> emb.forward(['a', 'b']) #doctest: +ELLIPSIS
    tensor(...)
    >>> emb.forward(['a', 'b']).shape
    torch.Size([10])
    """

    def __init__(self, alphabet: set, emb_size: int):
        """Create a random embedding dictionary.

        Arguments:
        * alphabet: set of symbols to embed (characters, words, POS tags, ...)
        * emb_size: embedding size (each symbol is mapped to a vector
            of size emb_size)
        """
        self.emb_size = emb_size
        self.enc = Encoding(alphabet)
        self.emb = nn.EmbeddingBag(self.enc.class_num, emb_size, mode='sum')

    def forward(self, syms: Iterator) -> TT:
        """Embed the given bag (sequence) of symbols and compute the sum.

        Returns:
            Single vector, which is the sum of the embeddings of the given
            symbols.
        """
        ixs = []
        for sym in syms:
            try:
                ixs.append(self.enc.encode(sym))
            except KeyError:
                pass
        if len(ixs) > 0:
            ix_tensor = torch.LongTensor(ixs).view(1, -1)
            return self.emb(ix_tensor)[0]
        else:
            return torch.zeros(self.emb_size)

    def params(self):
        """The list of parameters of the embedding dictionary."""
        return [self.emb.weight]
