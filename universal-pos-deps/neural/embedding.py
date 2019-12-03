from typing import Iterator

import torch
import torch.nn as nn

from neural.types import TT


# TODO: Implement this class
class Embedding(nn.Module):

    """A simple lookup table that stores embeddings
    of a fixed dictionary and size.

    >>> import torch
    >>> symset = set(['a', 'b', 'c'])
    >>> emb = Embedding(symset, emb_size=10)
    >>> emb('a') #doctest: +ELLIPSIS
    tensor(...)
    >>> emb('a').shape
    torch.Size([10])

    The `forwards` method allows to embed symbols in batches:
    >>> emb.forwards(['a', 'b', 'a', 'c']).shape
    torch.Size([4, 10])
    >>> (emb.forwards([['a', 'b', 'a', 'c']])[0] == emb('a')).all()
    True

    Embeddings are automatically registered as parameters of the model.
    In particular:
    >>> emb('a').requires_grad is True
    """

    def __init__(self, alphabet: set, emb_size: int):
        """Create a random embedding dictionary.

        Arguments:
        * alphabet: set of symbols to embed (characters, words, POS tags, ...)
        * emb_size: embedding size (each symbol is mapped to a vector
            of size emb_size)
        """
        # The following line is required in nn.Module subclasses
        super(Embedding, self).__init__()
        # Keep the embedding size
        self.emb_size = emb_size
        # TODO: implement the remaining of the initialization method,
        # using `nn.Embedding`

    # TODO: Implement this method
    def embedding_size(self) -> int:
        """Return the embedding size."""
        pass

    # TODO: Implement this method
    def forward(self, sym) -> TT:
        """Embed the given symbol."""
        pass

    def forwards(self, syms: Iterator) -> TT:
        """Embed the given sequence of symbols."""
        # TODO: This is a default implementation, which is correct but
        # sub-optimal in terms of speed.  You can improve it by applying
        # nn.Embedding to a batch of symbols at the same time.
        embs = []
        for sym in syms:
            embs.append(self.forward(sym))
        return torch.stack(embs)
