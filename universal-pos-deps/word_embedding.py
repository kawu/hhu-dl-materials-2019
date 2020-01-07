from typing import Iterable, Set

from abc import ABC, abstractmethod
import io

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


# DONE: Implement this as a part of Ex.~2.  HINT: use the
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


# TODO EX7: complete the implementation of this class
class FastText(WordEmbedder):
    """Module for fastText word embedding."""

    def __init__(self, file_path, limit: int = 10 ** 6):
        super(FastText, self).__init__()
        # Load vectors
        self._load_vectors(file_path, limit)

    def _load_vectors(self, fname, limit):
        # Code adapted from:
        # https://fasttext.cc/docs/en/english-vectors.html#format
        fast_file = io.open(fname, 'r', encoding='utf-8',
                            newline='\n', errors='ignore')
        _num, dim = map(int, fast_file.readline().split())
        # Store the embedding size
        self.emb_size = dim
        # Use .data dictionary to store the embeddings
        self.data = {}
        k = 0
        # Each subsequent line contains the word and the corresponding
        # embedding vector
        for line in fast_file:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            emb = list(map(float, tokens[1:]))
            assert len(emb) == dim
            self.data[word] = torch.tensor(emb, dtype=torch.float)
            # We only want to load a certain amout of most-frequent
            # embedding vectors, hence `break` below
            k += 1
            if k >= limit:
                break

    def forward(self, word: Word) -> TT:
        """Embed the given word."""
        # TODO EX7: implement this function
        pass

    def embedding_size(self):
        return self.emb_size
