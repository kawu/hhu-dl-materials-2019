from typing import Iterable, Dict

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
        self.obj_to_ix = {}  # type: Dict
        for ix, obj in enumerate(alphabet):
            self.obj_to_ix[obj] = ix
        # We use `len(self.obj_to_ix)` as a padding index, i.e.,
        # the index of the embedding vector fixed to 0 and which
        # is used to represent out-of-vocabulary words.
        self.padding_idx = len(self.obj_to_ix)
        # Create the nn.Embedding module; the vocabulary size
        # is set to `len(self.obj_to_ix)+1` because of the padding
        # index.
        self.emb = nn.Embedding(
            len(self.obj_to_ix)+1,
            emb_size,
            padding_idx=self.padding_idx
        )

    def embedding_size(self) -> int:
        """Return the embedding size."""
        return self.emb_size

    def forward(self, sym) -> TT:
        """Embed the given symbol."""
        try:
            ix = self.obj_to_ix[sym]
        except KeyError:
            # In case of out-of-vocabulary symbol/word,
            # use the padding index
            ix = self.padding_idx
        return self.emb(torch.tensor(ix, dtype=torch.long))

    def forwards(self, syms: Iterable) -> TT:
        """Embed the given sequence of symbols (word)."""
        ixs = []
        for sym in syms:
            try:
                ixs.append(self.obj_to_ix[sym])
            except KeyError:
                ixs.append(self.padding_idx)
        return self.emb(torch.LongTensor(ixs))

    def forwards_slow(self, syms: Iterable) -> TT:
        """Embed the given sequence of symbols."""
        # This is a default implementation, which is correct but
        # potentially sub-optimal in terms of speed.  We embed all
        # the symbols individually and then stack them together.
        # This means that, during training, the backward method of the
        # `torch.stack` method has to be used, which can be avoided by
        # embedding all the symbols together (see the `forwards` method
        # above).
        embs = []
        for sym in syms:
            embs.append(self.forward(sym))
        return torch.stack(embs)
