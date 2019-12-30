from typing import Sequence, Tuple, Iterator
# from abc import ABC, abstractmethod

from conllu import parse_incr

import torch.utils.data as data


# Input word
Word = str

# Part-of-speech tag
POS = str

# Annotated sentence: sequence of words paired with tuples
Sent = Sequence[Tuple[Word, POS]]


def load_data(file_path: str) -> Iterator[Sent]:
    """Load the dataset from a .conllu file."""
    with open(file_path, "r", encoding="utf-8") as data_file:
        for tok_list in parse_incr(data_file):
            sent = []
            for tok in tok_list:
                form = tok["form"]
                upos = tok["upostag"]
                # P8 -> Ex3: discard tokens which are not part of the
                # selected tokenization.  We assume that tokenization is done.
                if upos != '_':
                    sent.append((form, upos))
            yield sent


class PosDataSet(data.IterableDataset):
    """Abstract iterable over a POS dataset.

    The only purpose of this class is type annotation.  Use it to annotate
    functions/methods which expect POS dataset(s) as argument(s).
    """


class DiskPosDataSet(PosDataSet):
    """A POS dataset stored on a disk.

    Use this class to represent the POS dataset if you have limited memory
    and/or the dataset is particularly large.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __iter__(self):
        return load_data(self.file_path)


class MemPosDataSet(PosDataSet):
    """A POS dataset stored in memory.

    Use this class if speed is more important than memory usage.
    """

    def __init__(self, file_path: str, sort_by_len=False):
        self.data_set = list(load_data(file_path))
        if sort_by_len:
            self.data_set.sort(key=len, reverse=True)

    def __iter__(self):
        for elem in self.data_set:
            yield elem
