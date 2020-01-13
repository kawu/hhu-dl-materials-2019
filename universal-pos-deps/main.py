from typing import Sequence, Iterable, Set, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from neural.types import TT
from neural.training import train, batch_loader

from data import Word, POS, Sent
import data
from word_embedding import WordEmbedder, FastText


class PosTagger(nn.Module):
    """Simple POS tagger based on LSTM.

    * Each input word is embedded separately as an atomic object
    * LSTM is run over the word vector representations
    * The output "hidden" representations are used to predict the POS tags
    * Simple linear layer is used for scoring

    TODO: update the description considering dependency parsing.
    """

    def __init__(self,
                 word_emb: WordEmbedder, tagset: Set[POS], hid_size: int):
        super(PosTagger, self).__init__()
        # Keep the word embedder, so that it get registered
        # as a sub-module of the POS tagger.
        self.word_emb = word_emb
        # Keep the tagset
        self.tagset = tagset
        # We keep the size of the hidden layer equal to the embedding size
        # TODO EX8: set dropout-related in the LSTM
        self.lstm = nn.LSTM(
            self.word_emb.embedding_size(),
            hidden_size=hid_size,
            bidirectional=True
        )
        # We use the linear layer to score the embedding vectors
        self.linear_layer = nn.Linear(
            hid_size * 2,
            len(tagset)
        )

    ###########################################
    # Part I: scoring without batching
    ###########################################

    def embed(self, sent: Sequence[Word]) -> TT:
        """Embed and contextualize (using LSTM) the given sentence."""
        # Embed all the words and create the embedding matrix
        embs = self.word_emb.forwards(sent)
        # The first dimension should match the number of words
        assert embs.shape[0] == len(sent)
        # The second dimension should match the embedding size
        assert embs.shape[1] == self.word_emb.embedding_size()
        # Apply LSTM to word embeddings
        embs = embs.view(len(sent), 1, -1)
        ctx_embs, _ = self.lstm(embs)
        # Reshape back the contextualized embeddings
        ctx_embs = ctx_embs.view(len(sent), -1)
        # Return the resulting contextualized embeddings
        return ctx_embs

    def forward_pos(self, embs: TT) -> TT:
        """Calculate the POS score vectors for the individual words."""
        # The second dimension should match the scoring layer input size
        assert embs.shape[1] == self.linear_layer.in_features
        # Calculate the matrix with the scores
        scores = self.linear_layer(embs)
        # The second dimension should match the size of the tagset
        assert scores.shape[1] == len(self.tagset)
        # Finally, return the scores
        return scores

    # TODO: implement this function
    def forward_dep(self, embs: TT) -> TT:
        """Calculate the dependency score vectors for the individual words."""
        # Dummy implementation (TODO: add description)
        return torch.zeros(embs.shape[0], embs.shape[0]+1)

    def forward(self, sent: Sequence[Word]) -> Tuple[TT, TT]:
        """Calculate the score vectors for the individual words.

        The result is a pair of:
        * POS tagging-related scores (see `forward_pos`)
        * Dependency parsing-related scores (see `forward_dep`)
        """
        # Embed and contextualize the input words
        ctx_embs = self.embed(sent)
        # The first dimension should match the number of words
        assert ctx_embs.shape[0] == len(sent)
        # Calculate the POS scores
        pos_scores = self.forward_pos(ctx_embs)
        # Calculate the dependency scores
        dep_scores = self.forward_dep(ctx_embs)
        # Return the scores
        return pos_scores, dep_scores

    ###########################################
    # Part II: batching-enabled scoring
    ###########################################

    # def embeds(self, sents: Iterable[Sequence[Word]]) -> rnn.PackedSequence:
    #     """Embed and contextualize (using LSTM) the given batch."""
    #     # Embed all the sentences separately (we could further consider
    #     # embedding all the sentences at once, but this would require
    #     # creating a padded embedding (4-dimensional) tensor)
    #     embs = [
    #         self.word_emb.forwards(sent)
    #         for sent in sents
    #     ]
    #     # Pack embeddings as a packed sequence
    #     packed_embs = rnn.pack_sequence(embs, enforce_sorted=False)
    #     # The .data attribute of the packed sequence has the length
    #     # of the sum of the sentence lengths
    #     assert packed_embs.data.shape[0] == sum(len(semb) for semb in embs)
    #     # Apply LSTM to the packed sequence of word embeddings
    #     packed_hidden, _ = self.lstm(packed_embs)
    #     # The length of the .data attribute doesn't change (the cumulative
    #     # number of words in the batch does not change)
    #     assert packed_hidden.data.shape[0] == packed_embs.data.shape[0]
    #     # Return the resulting packed sequence with contextualized embeddings
    #     return packed_hidden

    # def forwards_pos(self, packed_hidden: rnn.PackedSequence) -> List[TT]:
    #     """Calculate the POS scores for the individual words."""
    #     # Each element of the .data attribute of the hidden packed sequence
    #     # should now match the input size of the linear scoring layer
    #     assert packed_hidden.data.shape[1] == self.linear_layer.in_features
    #     # Apply the linear layer to each element of `packed_hidden.data`
    #     # individually
    #     scores_data = self.linear_layer(packed_hidden.data)
    #     # Recreate a packed sequence out of the resulting scoring data;
    #     # this is somewhat low-level and not really recommended by PyTorch
    #     # documentation -- do you know a better way?
    #     packed_scores = rnn.PackedSequence(
    #         scores_data,
    #         batch_sizes=packed_hidden.batch_sizes,
    #         sorted_indices=packed_hidden.sorted_indices,
    #         unsorted_indices=packed_hidden.unsorted_indices
    #     )
    #     # Pad the resulting packed sequence
    #     padded_scores, padded_len = rnn.pad_packed_sequence(
    #         packed_scores, batch_first=True)
    #     # Convert the padded representation to a list of score tensors
    #     scores = []
    #     for sco, n in zip(padded_scores, padded_len):
    #         scores.append(sco[:n])
    #     # Finally, return the scores
    #     return scores

    # def forwards(self, sents: Iterable[Sequence[Word]]) -> List[Tuple[TT, TT]]:
    #     """Calculate the score vectors for the individual words.

    #     Batching-enabled version of `forwards`.
    #     """
    #     # Embed and contextualize the entire batch
    #     packed_hidden = self.embeds(sents)
    #     # Calculate the POS-related scores
    #     pos_scores = self.forwards_pos(packed_hidden)
    #     # Return the scores
    #     return zip(pos_scores, None)

    def forwards(self, sents):
        ys = []
        for sent in sents:
            ys.append(self.forward(sent))
        return ys

    ###########################################
    # Part III: tagging (evaluation mode)
    ###########################################

    def tag(self, sent: Sequence[Word]) -> Sequence[POS]:
        """Predict the POS tags in the given sentence."""
        return list(self.tags([sent]))[0]

    def tags(self, batch: Sequence[Sequence[Word]]) -> Iterable[Sequence[POS]]:
        """Predict the POS tags in the given batch of sentences.

        A variant of the `tag` method which works on a batch of sentences.
        """
        # TODO EX8: make sure that dropout is not applied when tagging!
        # TODO: does it make sense to use `tag` as part of training?
        with torch.no_grad():
            # POS tagging is to be carried out based on the resulting scores
            scores_batch, _ = zip(*self.forwards(batch))
            for scores, sent in zip(scores_batch, batch):
                # Create a list for predicted POS tags
                predictions = []
                # For each word, we must select the POS tag corresponding to
                # the index with the highest score
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
                yield predictions


def accuracy(
        tagger: PosTagger, data_set: Iterable[Sent], batch_size=64) -> float:
    """Calculate the accuracy of the model on the given dataset.

    The accuracy is defined as the percentage of the words in the data_set
    for which the model predicts the correct POS tag.
    """
    k, n = 0., 0.
    # We load the dataset in batches to speed the calculation up
    for batch in batch_loader(data_set, batch_size=batch_size):
        # Calculate the input batch
        inputs = []
        for sent in batch:
            # Split the sentence into input words and POS tags
            words = list(map(lambda tok: tok.word, sent))
            inputs.append(list(words))
        # Tag all the sentences
        predictions = tagger.tags(inputs)
        # Process the predictions and compare with the gold POS tags
        for sent, pred_poss in zip(batch, predictions):
            # Split the sentence into input words and POS tags
            gold_tags = map(lambda tok: tok.upos, sent)
            for (pred_pos, gold_pos) in zip(pred_poss, gold_tags):
                if pred_pos == gold_pos:
                    k += 1.
                n += 1.
    return k / n


def pos_loss(tagger: PosTagger, data_set: Iterable[Sent]) -> TT:
    """The POS tagging-related cross entropy loss over the given dataset."""
    # Create two lists for target indices (corresponding to POS tags we
    # want our mode to predict) and the actually predicted scores.
    target_ixs = []     # type: List[int]
    inputs = []         # type: List[Sequence[Word]]
    # Loop over the dataset in order to determine the target POS tags
    # and the predictions
    for sent in data_set:
        # Unzip the sentence into a (list of words, list of target POS tags)
        words = list(map(lambda tok: tok.word, sent))
        gold_tags = map(lambda tok: tok.upos, sent)
        # DONE: Determine the target POS tag indices and update `target_ixs`
        for tag in gold_tags:
            # Determine the index corresponding to the POS gold_tag
            ix = list(tagger.tagset).index(tag)
            # Append it to the target list
            target_ixs.append(ix)
        # Append the new sentence to the inputs list
        inputs.append(words)
    # Calculate the scores in a batch and concat them
    pos_scores, _ = zip(*tagger.forwards(inputs))
    pos_scores = torch.cat(pos_scores)
    # Convert the target indices to a tensor
    target_ixs = torch.LongTensor(target_ixs)
    # Make sure the dimensions match
    assert target_ixs.shape[0] == pos_scores.shape[0]
    # Assert that target_ixs is a vector (1d tensor)
    assert target_ixs.dim() == 1
    # The second dimension of the predicted scores
    # should correspond to the size of the tagset:
    assert pos_scores.shape[1] == len(tagger.tagset)
    # Calculate the loss and return it
    loss = nn.CrossEntropyLoss()
    return loss(pos_scores, target_ixs)


# Training dataset
train_set = data.MemPosDataSet(
    "UD_English-ParTUT/en_partut-ud-train.conllu",
    sort_by_len=True
)

# Development dataset
dev_set = data.MemPosDataSet("UD_English-ParTUT/en_partut-ud-dev.conllu")

# Size stats
print("Train size:", len(list(train_set)))
print("Dev size:", len(list(dev_set)))

# Determine the set of words in the dataset
word_set = set(
    tok.word
    for sent in train_set
    for tok in sent
)

# Number of words
print("Number of words:", len(word_set))

# Determine the POS tagset
tagset = set(
    tok.upos
    for sent in train_set
    for tok in sent
)

# Tagset
print("Tagset:", tagset)

# Create the word embedding module
# word_emb = AtomicEmbedder(word_set, 10)
word_emb = FastText("wiki-news-300d-1M-subword-selected.vec")

# Create the tagger
tagger = PosTagger(word_emb, tagset, hid_size=10)

# Train the model (see `train` in `neural/training`)
train(
    tagger, train_set, dev_set,
    pos_loss, accuracy,
    epoch_num=25,
    learning_rate=0.01,
    report_rate=5
)
