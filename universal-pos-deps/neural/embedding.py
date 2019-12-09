from typing import Iterator

import torch
import torch.nn as nn

from neural.types import TT


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
    >>> assert (emb.forwards(['a', 'b', 'a', 'c'])[0] == emb('a')).all()

    Embeddings are automatically registered as parameters of the model.
    In particular:
    >>> emb('a').requires_grad is True
    True
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
        # Create the mapping from alphabet/vocabulary to indices
        self.obj_to_ix = {}
        for ix, obj in enumerate(alphabet):
            self.obj_to_ix[obj] = ix
        # Create the nn.Embedding module
        self.emb = nn.Embedding(len(self.obj_to_ix), emb_size)

    def embedding_size(self) -> int:
        """Return the embedding size."""
        return self.emb_size

    def forward(self, sym) -> TT:
        """Embed the given symbol."""
        try:
            ix = self.obj_to_ix[sym]
            return self.emb(torch.tensor(ix, dtype=torch.long))
        except KeyError:
            return torch.zeros(self.emb_size)

    def forwards(self, syms: Iterator) -> TT:
        """Embed the given sequence of symbols."""
        # TODO: This is a default implementation, which is correct but
        # potentially sub-optimal in terms of speed.  You can improve it
        # by applying nn.Embedding to a batch of symbols at the same time.
        embs = []
        for sym in syms:
            embs.append(self.forward(sym))
        return torch.stack(embs)
