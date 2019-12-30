from typing import Iterable, Set

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from data import Word
from neural.types import TT
from neural.embedding import Embedding


class WordEmbedder(ABC, nn.Module):

    @abstractmethod
    def forward(self, word: Word) -> TT:
        """Embed the given word."""
        pass

    # @abstractmethod
    def forwards(self, words: Iterable[Word]) -> TT:
        """Embed the given words."""
        # Default implementation.  Re-implement for speed
        # in a sub-class.
        return torch.stack([self.forward(word) for word in words])

    @abstractmethod
    def embedding_size(self) -> int:
        """Return the size of the embedding vectors."""
        pass


# TODO: Implement this as a part of Ex.~2.  HINT: use the
# Embedding class implemented in neural/embedding.py.
class AtomicEmbedder(WordEmbedder):
    """Word embedding class which considers each word as an atomic entity.
    Put differently, each word receives its own embedding vector.

    For example, let's take a small vocabulary:
    >>> vocab = set(["cat", "cats", "Cat"])

    And create a case-insensitive word embedder:
    >>> emb = AtomicEmbedder(vocab, emb_size=10, case_insensitive=True)

    Retrieve the embedding for "cat":
    >>> cat_emb = emb("cat")
    >>> cat_emb.shape
    torch.Size([10])

    Given that the embedder is case-insensitive, it should give the same
    embedding to both "cat" and "Cat":
    >>> assert (emb("cat") == emb("Cat")).all()

    You can apply it to a sequence of words at the same time. You don't
    have to implement anything to obtain this behavior, since the default
    implementation of `forwards` is provided by the abstract WordEmbedder
    class:
    >>> many_embs = emb.forwards(["cat", "cats"])
    >>> assert (many_embs[0] == cat_emb).all()

    Each embedding should be accounted for during training.
    In particular:
    >>> assert emb("cat").requires_grad is True

    For out-of-vocabulary words, the embedder should return 0:
    >>> assert (emb("dog") == 0).all()
    """

    def __init__(self, vocab: Set[Word], emb_size: int,
                 case_insensitive=False):
        """Create the word embedder for the given vocabulary.

        Arguments:
            vocab: vocabulary of words to embed
            emb_size: the size of embedding vectors
            case_insensitive: should the embedder be case-insensitive?
        """
        # The following line is required in each custom neural Module.
        super(AtomicEmbedder, self).__init__()
        # Keep info about the case sensitivity
        self.case_insensitive = case_insensitive
        # Calculate the modified vocabulary
        vocab = set(self.preprocess(x) for x in vocab)
        # Initialize the generic embedding module
        self.emb = Embedding(vocab, emb_size)

    def preprocess(self, word: Word) -> Word:
        """Preprocessing function"""
        # We use word.lower() to make the embedder case-insensitive
        if self.case_insensitive:
            return word.lower()
        else:
            return word

    def forward(self, word: Word) -> TT:
        """Embed the given word as a vector."""
        return self.emb(self.preprocess(word))

    def forwards(self, words: Iterable[Word]) -> TT:
        """Embed the given sequence of words."""
        # This is faster than the default implementation (see the
        # `WordEmbedder` class) because it relies on the more
        # performant `self.emb.fowards` method.
        return self.emb.forwards(map(self.preprocess, words))

    def embedding_size(self) -> int:
        """Return the embedding size of the word embedder."""
        return self.emb.embedding_size()
