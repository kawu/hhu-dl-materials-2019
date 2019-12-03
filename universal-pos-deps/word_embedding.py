from typing import Iterator

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from data import Word
from neural.types import TT


class WordEmbedder(ABC, nn.Module):

    @abstractmethod
    def forward(self, word: Word) -> TT:
        """Embed the given word."""
        pass

    # @abstractmethod
    def forwards(self, words: Iterator[Word]) -> TT:
        """Embed the given words."""
        # Default implementation.  Re-implement for speed
        # in a sub-class.
        return torch.stack(self.forward(word) for word in words)

    @abstractmethod
    def embedding_size(self) -> int:
        """Return the size of the embedding vectors."""
        pass


# TODO: Implement this as a part of Ex.~2.  HINT: use the
# Embedding class implemented in neural/embedding.py.
class AtomEmbedder(WordEmbedder):
    """Word embedding class which considers each word as an atomic entity.
    Put differently, each word receives its own embedding vector.

    For example, let's take a small vocabulary:
    >>> vocab = set(["cat", "cats", "Cat"])

    And create a case-insensitive word embedder:
    >>> emb = AtomicEmbedder(vocab, emb_size=10, case_insensitive=True)

    Retrieve the embedding for "cat":
    >>> cat_emb = emb("cat")
    >>> cat_emb.shape
    torch.Shape([10])

    Given that the embedder is case-insensitive, it should give the same
    embedding to both "cat" and "Cat":
    >>> assert (emb("cat") == emb("Cat")).all()

    You can apply it to a sequence of words at the same time. You don't
    have to implement anything to obtain this behavior, since the default
    implementation of `forwards` is provided by the abstract WordEmbedder
    class.
    >>> many_embs = emb.forwards(["cat", "Cat"])
    >>> assert (many_embs[0] == cat_emb).all()

    And of course each embedding should be accounted for during training.
    In particular:
    >>> assert emb("cat").requires_grad is True
    """

    # TODO: implement the initialization method

    # TODO: implement this method:
    def forward(self, word: Word) -> TT:
        pass

    # TODO: implement this method:
    def embedding_size(self) -> int:
        pass
