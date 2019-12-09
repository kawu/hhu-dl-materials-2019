from typing import Sequence, Iterable, Set

import torch.nn as nn

from neural.types import TT

from data import Word, POS, Sent
import data
from word_embedding import WordEmbedder, AtomicEmbedder


# TODO: Implement this class
class PosTagger(nn.Module):
    """Simple POS tagger based on LSTM.

    * Each input word is embedded separately as an atomic object
    * LSTM is run over the word vector representations
    * The output "hidden" representations are used to predict the POS tags
    * Simple linear layer is used for scoring
    """

    # TODO: implement this method
    def __init__(self, word_emb: WordEmbedder, tagset: Set[POS]):
        pass

    # TODO: implement this method
    def forward(self, sent: Sequence[Word]) -> TT:
        """Calculate the score vectors for the individual words."""
        # TODO: Embed all the words and create the embedding matrix
        embs = None
        # The first dimension should match the number of words
        assert embs.shape[0] == len(sent)
        # TODO: The second dimension should match...?
        assert embs.shape[1] == -1
        # TODO: Calculate the matrix with the scores
        scores = None
        # The first dimension should match the number of words
        assert scores.shape[0] == len(sent)
        # TODO: The second dimension should match...?
        assert scores.shape[1] == -1
        # Finally, return the scores
        return scores

    # TODO: implement this method
    def tag(self, sent: Sequence[Word]) -> Sequence[POS]:
        """Predict the POS tags in the given sentence."""
        # POS tagging is to be carried out based on the resulting scores
        scores = self.forward(sent)
        # Create a list for predicted POS tags
        predictions = []
        # For each word, we must select the POS tag corresponding to the
        # index with the highest score
        for score_vect in scores:
            # TODO: determine the position with the highest score
            ix = None
            # TODO: assert the index is within the range of POS tag indices
            assert 0 <= ix < None
            # TODO: determine the corresponding POS tag
            pos = None
            # Append the new prediction
            predictions.append(pos)
        # We should have as many predicted POS tags as input words
        assert len(sent) == len(predictions)
        # Return the predicted POS tags
        return predictions


def accuracy(tagger: PosTagger, data_set: Iterable[Sent]) -> float:
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


# TODO: implement this function
def total_loss(tagger: PosTagger, data_set: Iterable[Sent]) -> TT:
    """Calculate the total total, cross entropy loss over the given dataset."""
    # Create two lists for target indices (corresponding to POS tags we
    # want our mode to predict) and the actually predicted scores.
    target_ixs = []
    pred_scores = []
    # Loop over the dataset in order to determine the target POS tags
    # and the predictions
    for sent in data_set:
        # Unzip the sentence into a (list of words, list of target POS tags)
        (words, gold_tags) = zip(*sent)
        # TODO: Determine the target POS tag indices and update `target_ixs`
        # TODO: Determine the predicted scores and update `pred_scores`
    # TODO: Convert the target indices and the predictions to tensors
    pass
    # Make sure the dimensions match
    assert target_ixs.shape[0] == pred_scores.shape[0]
    # TODO: In particular, the second dimension of the predicted scores
    # should correspond to the size of the tagset, i.e.?
    assert pred_scores.shape[1] == None
    # TODO: Calculate the loss and return it
    pass


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

# Create the word embedding module
word_emb = AtomicEmbedder(word_set, 50)

# Create the tagger
tagger = PosTagger(word_emb, tagset)

# TODO: train the model (see `train` in `neural/training`)
