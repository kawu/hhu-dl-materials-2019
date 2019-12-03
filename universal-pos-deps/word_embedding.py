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
