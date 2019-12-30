from typing import Sequence, Iterable, Set, List

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from neural.types import TT
from neural.training import train

from data import Word, POS, Sent
import data
from word_embedding import WordEmbedder, AtomicEmbedder


# DONE: Implement this class
class PosTagger(nn.Module):
    """Simple POS tagger based on LSTM.

    * Each input word is embedded separately as an atomic object
    * LSTM is run over the word vector representations
    * The output "hidden" representations are used to predict the POS tags
    * Simple linear layer is used for scoring
    """

    # DONE EX6: adapt this method to use LSTM
    def __init__(self, word_emb: WordEmbedder, tagset: Set[POS]):
        super(PosTagger, self).__init__()
        # Keep the word embedder, so that it get registered
        # as a sub-module of the POS tagger.
        self.word_emb = word_emb
        # Keep the tagset
        self.tagset = tagset
        # We use the linear layer to score the embedding vectors
        # TODO EX6: account for LSTM
        # DONE: we keep the size of the hidden layer equal to
        # the embedding size
        self.linear_layer = nn.Linear(
            self.word_emb.embedding_size(),
            len(tagset)
        )
        # # To normalize the output of the linear layer
        # # (do we need it?)
        # self.normalizer = nn.Sigmoid()
        # DONE EX6: add LSTM submodule
        self.lstm = nn.LSTM(
            self.word_emb.embedding_size(),
            self.word_emb.embedding_size()
        )

    # DONE EX6: adapt this method to use LSTM
    def forward(self, sent: Sequence[Word]) -> TT:
        """Calculate the score vectors for the individual words."""
        # Embed all the words and create the embedding matrix
        embs = self.word_emb.forwards(sent)
        # The first dimension should match the number of words
        assert embs.shape[0] == len(sent)
        # The second dimension should match the embedding size
        assert embs.shape[1] == self.word_emb.embedding_size()
        # DONE EX6: apply LSTM to word embeddings
        embs = embs.view(len(sent), 1, -1)
        ctx_embs, _ = self.lstm(embs)
        # Reshape back the contextualized embeddings
        ctx_embs = ctx_embs.view(len(sent), -1)
        # Calculate the matrix with the scores
        scores = self.linear_layer(ctx_embs)
        # The first dimension should match the number of words
        assert scores.shape[0] == len(sent)
        # The second dimension should match the size of the tagset
        assert scores.shape[1] == len(self.tagset)
        # Finally, return the scores
        return scores

    def forwards(self, sents: Iterable[Sequence[Word]]) -> List[TT]:
        """Calculate the score vectors for the individual words."""
        # Embed all the sentences separately (we could further consider
        # embedding all the sentences at once, but this would require
        # creating a padded embedding (4-dimensional) tensor)
        embs = [
            self.word_emb.forwards(sent)
            for sent in sents
        ]
        # Pack embeddings as a packed sequence
        packed_embs = rnn.pack_sequence(embs, enforce_sorted=False)
        # Apply LSTM to the packed sequence of word embeddings
        packed_hidden, _ = self.lstm(packed_embs)
        # Each element of the .data attribute of the resulting hidden
        # packed sequence should now match the input size of the linear
        # scoring layer
        assert packed_hidden.data.shape[1] == self.linear_layer.in_features
        # Apply the linear layer to each element of `packed_hidden.data`
        # individually
        scores_data = self.linear_layer(packed_hidden.data)
        # Recreate a packed sequence out of the resulting scoring data;
        # this is somewhat low-level and not really recommended by PyTorch
        # documnetation -- do you know a better way?
        packed_scores = rnn.PackedSequence(
            scores_data,
            batch_sizes=packed_hidden.batch_sizes,
            sorted_indices=packed_hidden.sorted_indices,
            unsorted_indices=packed_hidden.unsorted_indices
        )
        # Pad the resulting packed sequence
        padded_scores, padded_len = rnn.pad_packed_sequence(
            packed_scores, batch_first=True)
        # Convert the padded representation to a list of score tensors
        scores = []
        for sco, n in zip(padded_scores, padded_len):
            scores.append(sco[:n])
        # Finally, return the scores
        return scores

    # DONE (modulo testing): implement this method
    def tag(self, sent: Sequence[Word]) -> Sequence[POS]:
        """Predict the POS tags in the given sentence."""
        # TODO: does it make sense to use `tag` as part of training?
        with torch.no_grad():
            # POS tagging is to be carried out based on the resulting scores
            scores = self.forward(sent)
            # Create a list for predicted POS tags
            predictions = []
            # For each word, we must select the POS tag corresponding to the
            # index with the highest score
            for score_vect in scores:
                # Determine the position with the highest score
                _, ix = torch.max(score_vect, 0)
                # Make sure `ix` is a 0d tensor
                assert ix.dim() == 0
                ix = ix.item()
                # Assert the index is within the range of POS tag indices
                assert 0 <= ix < len(self.tagset)
                # Determine the corresponding POS tag
                pos = list(self.tagset)[ix]
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


# DONE: implement this function
def total_loss(tagger: PosTagger, data_set: Iterable[Sent]) -> TT:
    """Calculate the total total, cross entropy loss over the given dataset."""
    # Create two lists for target indices (corresponding to POS tags we
    # want our mode to predict) and the actually predicted scores.
    target_ixs = []     # type: List[int]
    inputs = []         # type: List[Sequence[Word]]
    # Loop over the dataset in order to determine the target POS tags
    # and the predictions
    for sent in data_set:
        # Unzip the sentence into a (list of words, list of target POS tags)
        (words, gold_tags) = zip(*sent)
        # DONE: Determine the target POS tag indices and update `target_ixs`
        for tag in gold_tags:
            # Determine the index corresponding to the POS gold_tag
            ix = list(tagger.tagset).index(tag)
            # Append it to the target list
            target_ixs.append(ix)
        # Append the new sentence to the inputs list
        inputs.append(words)
    # Calculate the scores in a batch and concat them
    pred_scores = torch.cat(tagger.forwards(inputs))
    # Convert the target indices to a tensor
    target_ixs = torch.LongTensor(target_ixs)
    # Make sure the dimensions match
    assert target_ixs.shape[0] == pred_scores.shape[0]
    # Assert that target_ixs is a vector (1d tensor)
    assert target_ixs.dim() == 1
    # The second dimension of the predicted scores
    # should correspond to the size of the tagset:
    assert pred_scores.shape[1] == len(tagger.tagset)
    # DONE: Calculate the loss and return it
    loss = nn.CrossEntropyLoss()
    return loss(pred_scores, target_ixs)


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
word_emb = AtomicEmbedder(word_set, 10)

# Create the tagger
tagger = PosTagger(word_emb, tagset)

# DONE: train the model (see `train` in `neural/training`)
train(
    tagger, train_set, dev_set,
    total_loss, accuracy,
    epoch_num=25,
    learning_rate=0.01,
    report_rate=5
)
