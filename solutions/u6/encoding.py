from typing import Iterable


class Encoding:

    """A class which represents a mapping between target classes and
    the corresponding vector representations.

    >>> classes = ["English", "German", "French"]
    >>> enc = Encoding(classes)
    >>> assert "English" == enc.decode(enc.encode("English"))
    >>> assert "German" == enc.decode(enc.encode("German"))
    >>> assert "French" == enc.decode(enc.encode("French"))
    >>> set(range(3)) == set(enc.encode(cl) for cl in classes)
    True
    >>> for cl in classes:
    ...     ix = enc.encode(cl)
    ...     assert 0 <= ix <= enc.class_num
    ...     assert cl == enc.decode(ix)
    """

    def __init__(self, classes: Iterable[str]):
        class_set = set(cl for cl in classes)
        self.class_num = len(class_set)
        self.class_to_ix = {}
        self.ix_to_class = {}
        for (ix, cl) in enumerate(class_set):
            self.class_to_ix[cl] = ix
            self.ix_to_class[ix] = cl

    def encode(self, cl: str) -> int:
        return self.class_to_ix[cl]

    def decode(self, ix: int) -> str:
        return self.ix_to_class[ix]
