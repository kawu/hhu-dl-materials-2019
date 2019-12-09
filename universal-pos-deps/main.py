from typing import Sequence, Iterator

import torch.nn as nn

from data import Word, POS, Sent
import data


# TODO: Implement this class
class PosTagger(nn.Module):
    """Simple POS tagger based on LSTM.

    * Each input word (token) is embedded separately as an atomic object
    * LSTM is run over the word vector representations
    * The output "hidden" representations are used to predict the POS tags
    * Simple linear layer is used for scoring
    """

    def tag(self, sent: Sequence[Word]) -> Sequence[POS]:
        """Predict the POS tags in the given sentence."""
        pass


def accuracy(tagger: PosTagger, data_set: Iterator[Sent]) -> float:
    """Calculate the accuracy of the model on the given dataset.

    The accuracy is defined as the percentage of the words in the data_set
    for which the model predicts the correct POS tag.
    """
    k, n = 0., 0.
    for sent in data_set:
        # Split the sentence into input words and POS tags
        words, gold_poss = zip(*sent)
        # Predict the POS tags using the model
        pred_poss = tagger.tag(words)
        for (pred_pos, gold_pos) in zip(pred_poss, gold_poss):
            if pred_pos == gold_pos:
                k += 1.
            n += 1.
    return k / n


# Training dataset
train_set = data.MemPosDataSet("UD_English-ParTUT/en_partut-ud-train.conllu")

# Development dataset
dev_set = data.MemPosDataSet("UD_English-ParTUT/en_partut-ud-dev.conllu")

# Size stats
print("Train size:", len(list(train_set)))
print("Dev size:", len(list(dev_set)))

# Determine the set of words in the dataset
word_set = set(
    word
    for sent in train_set
    for (word, _pos) in sent
)

# Number of words
print("Number of words:", len(word_set))

# Determine the POS tagset
tagset = set(
    pos
    for sent in train_set
    for (_word, pos) in sent
)

# Tagset
print("Tagset:", tagset)

# TODO: finish the implementation of the model:
# * Create the word embedding neural module
# * Calculate the POS tagset
# * Create the POS tagger (PosTagger)
# * Train the tagger on the training data
