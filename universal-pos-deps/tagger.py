
from typing import Sequence, Iterable, Set, List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

from neural.types import TT
from neural.training import batch_loader

from data import Word, POS, Head, Sent
from word_embedding import WordEmbedder


class Tagger(nn.Module):
    """LSTM-based POS tagger and dependency parser.

    * Each input word is embedded separately as an atomic object
    * LSTM is applied to the sequence of the word vector representations
    * The output "hidden" representations are used to predict the POS tags;
      To this end, simple linear layer is used for scoring
    * For each two words in the sentence, their hidden representations
      are matched to provide a score of one word being the head of the
      other word; then, for each word, the word with the highest score
      is selected as its head
    """

    def __init__(self,
                 word_emb: WordEmbedder, tagset: Set[POS], hid_size: int):
        super(Tagger, self).__init__()
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
            bidirectional=True,
            num_layers=2,
            dropout=0.2
        )
        # We use the linear layer to score the embedding vectors
        self.linear_layer = nn.Linear(
            hid_size * 2,
            len(tagset)
        )
        # Dependent represnetation
        self.dep_repr = nn.Linear(
            hid_size * 2,
            hid_size * 2
        )
        # Head represnetation
        self.hed_repr = nn.Linear(
            hid_size * 2,
            hid_size * 2
        )
        # Representation of the dummy root
        self.root_repr = nn.Parameter(torch.zeros(hid_size*2))

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
        # This is a dummy implementation, in which the score between each
        # two words is set to be 0; the task is two calculate meaningful
        # scores
        sent_len = embs.shape[0]
        # Calculate the dependent representations
        D = self.dep_repr(embs)
        # Calculate the head representations
        H = self.hed_repr(embs)
        # Add the root dummy vector
        H_r = torch.cat([self.root_repr.view(1, -1), H], dim=0)
        # Calculate the resulting scores
        scores = torch.mm(D, H_r.t())
        # Make sure that the shape is correct
        assert scores.dim() == 2
        assert scores.shape[0] == sent_len
        assert scores.shape[1] == sent_len + 1
        # Return the scores
        return scores

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

    def embeds(self, sents: Iterable[Sequence[Word]]) -> rnn.PackedSequence:
        """Embed and contextualize (using LSTM) the given batch."""
        # Embed all the sentences separately (we could further consider
        # embedding all the sentences at once, but this would require
        # creating a padded embedding (4-dimensional) tensor)
        embs = [
            self.word_emb.forwards(sent)
            for sent in sents
        ]
        # Pack embeddings as a packed sequence
        packed_embs = rnn.pack_sequence(embs, enforce_sorted=False)
        # The .data attribute of the packed sequence has the length
        # of the sum of the sentence lengths
        assert packed_embs.data.shape[0] == sum(len(semb) for semb in embs)
        # Apply LSTM to the packed sequence of word embeddings
        packed_hidden, _ = self.lstm(packed_embs)
        # The length of the .data attribute doesn't change (the cumulative
        # number of words in the batch does not change)
        assert packed_hidden.data.shape[0] == packed_embs.data.shape[0]
        # Return the resulting packed sequence with contextualized embeddings
        return packed_hidden

    def forwards_pos(self, packed_hidden: rnn.PackedSequence) -> List[TT]:
        """Calculate the POS scores for the individual words."""
        # Each element of the .data attribute of the hidden packed sequence
        # should now match the input size of the linear scoring layer
        assert packed_hidden.data.shape[1] == self.linear_layer.in_features
        # Apply the linear layer to each element of `packed_hidden.data`
        # individually
        scores_data = self.linear_layer(packed_hidden.data)
        # Recreate a packed sequence out of the resulting scoring data;
        # this is somewhat low-level and not really recommended by PyTorch
        # documentation -- do you know a better way?
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

    # TODO: optimize this function; currently, it relies on `forward_dep`,
    # which is not batching-enabled
    def forwards_dep(self, packed_hidden: rnn.PackedSequence) -> List[TT]:
        """Calculate the dependency scores for the individual words."""
        # Convert packed representation to a padded representation
        padded_hidden, padded_len = rnn.pad_packed_sequence(
            packed_hidden, batch_first=True
        )
        # Split the padded representation into sentences and calculate the
        # dependency scores using `forward_dep`
        scores = []
        for hidden, n in zip(padded_hidden, padded_len):
            hidden = hidden[:n]  # Remove padding
            scores.append(self.forward_dep(hidden))
        # Finally, return the scores
        return scores

    def forwards(self, sents: Iterable[Sequence[Word]]) -> List[Tuple[TT, TT]]:
        """Calculate the score vectors for the individual words.

        Batching-enabled version of `forwards`.
        """
        # Embed and contextualize the entire batch
        packed_hidden = self.embeds(sents)
        # Calculate the scores
        pos_scores = self.forwards_pos(packed_hidden)
        dep_scores = self.forwards_dep(packed_hidden)
        # Return the scores
        return list(zip(pos_scores, dep_scores))

    ###########################################
    # Part III: tagging (evaluation mode)
    ###########################################

    def tag(self, sent: Sequence[Word]) -> Sequence[Tuple[POS, Head]]:
        """Predict the POS tags and dependency heads in the given sentence."""
        return list(self.tags([sent]))[0]

    def tags(self, batch: Sequence[Sequence[Word]]) \
            -> Iterable[List[Tuple[POS, Head]]]:
        """Predict the POS tags and dependency heads in the given batch."""
        # TODO: does it make sense to use `tag` as part of training?
        with torch.no_grad():
            # Turn evaluation mode on
            self.eval()
            # POS tagging is to be carried out based on the resulting scores
            pos_scores_batch, dep_scores_batch = zip(*self.forwards(batch))
            # Turn evaluation mode off
            self.train()
            # print("train mode is on:", self.training)
            for pos_scores, dep_scores, sent in zip(
                    pos_scores_batch, dep_scores_batch, batch):
                # Predict POS tags and dependency heads
                pos_preds = self.predict_pos_tags(pos_scores)
                dep_preds = self.predict_heads(dep_scores)
                # We should have as many predicted POS tags and dependency
                # heads as input words
                assert len(sent) == len(pos_preds) == len(dep_preds)
                # Return the predicted POS tags
                yield list(zip(pos_preds, dep_preds))

    def predict_pos_tags(self, pos_scores: TT) -> List[POS]:
        """Predict POS tags given POS-related scores (single sentence)."""
        # Sentence length
        sent_len = len(pos_scores)
        # List for POS tags
        pos_predictions = []
        # For each word, we must select the POS tag corresponding to
        # the index with the highest score
        for score_vect in pos_scores:
            # Determine the position with the highest score
            ix = torch.argmax(score_vect).item()
            # Assert the index is within the range of POS tag indices
            assert 0 <= ix < len(self.tagset)
            # Determine the corresponding POS tag
            pos = list(self.tagset)[ix]
            # Append the new prediction
            pos_predictions.append(pos)
        # We should have as many predicted POS tags as input words
        assert sent_len == len(pos_predictions)
        # Return the predicted POS tags
        return pos_predictions

    def predict_heads(self, head_scores: TT) -> List[Head]:
        """Predict dependencies based on the head scores (single sentence)."""
        # Sentence length
        sent_len = len(head_scores)
        # List for dependency heads
        head_predictions = []
        for score_vect in head_scores:
            # Determine the position with the highest score
            ix = torch.argmax(score_vect).item()
            # Assert the index is within the range of the possible head indices
            assert 0 <= ix <= sent_len
            # Append the new prediction
            head_predictions.append(ix)
        # We should have as many predicted POS tags as input words
        assert sent_len == len(head_predictions)
        # Return the predictions
        return head_predictions


def pos_accuracy(
        tagger: Tagger, data_set: Iterable[Sent], batch_size=64) -> float:
    """Calculate the POS tagging accuracy of the model on the given dataset.

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
        for sent, pred_poss_deps in zip(batch, predictions):
            # Extract POS tags
            pred_poss, _ = zip(*pred_poss_deps)
            # Split the sentence into input words and POS tags
            gold_tags = map(lambda tok: tok.upos, sent)
            for (pred_pos, gold_pos) in zip(pred_poss, gold_tags):
                if pred_pos == gold_pos:
                    k += 1.
                n += 1.
    return k / n


def dep_accuracy(
        tagger: Tagger, data_set: Iterable[Sent], batch_size=64) -> float:
    """Calculate the unlabeled attachment score (UAS) on the given dataset.

    UAS is defined as the percentage of the words in the data_set
    for which the model predicts the correct dependency head.
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
        for sent, pred_poss_deps in zip(batch, predictions):
            # Extract dependency heads
            _, pred_heads = zip(*pred_poss_deps)
            # Split the sentence into input words and POS tags
            gold_heads = map(lambda tok: tok.head, sent)
            for (pred_pos, gold_pos) in zip(pred_heads, gold_heads):
                if pred_pos == gold_pos:
                    k += 1.
                n += 1.
    return k / n


def pos_loss(tagger: Tagger, data_set: Iterable[Sent]) -> TT:
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


def total_loss_simple(tagger: Tagger, data_set: Iterable[Sent]) -> TT:
    """Calculate the total cross entropy loss over the given dataset.

    The total loss is defined as a sum of the POS tagging-related loss and
    the dependency parsing-related loss.

    See also `total_loss` for a more optimized version.
    """

    #########################################################
    # Calculate the inputs and the target outputs
    #########################################################

    # Create a list for input sentences
    inputs = []          # type: List[Sequence[Word]]

    # Create two lists for target indices (POS tags and head indices)
    target_pos_ixs = []  # type: List[int]
    target_heads = []    # type: List[TT]

    # Loop over the dataset to determine target indices and input words
    for sent in data_set:
        # Extract the input words, gold POS tags, and gold dependency heads
        words = map(lambda tok: tok.word, sent)
        gold_tags = map(lambda tok: tok.upos, sent)
        gold_heads = map(lambda tok: tok.head, sent)
        # Append the new sentence to the inputs list
        inputs.append(list(words))
        # Determine the target POS tag indices
        for tag in gold_tags:
            # Determine the index corresponding to the gold POS tag
            ix = list(tagger.tagset).index(tag)
            # Append it to the target list
            target_pos_ixs.append(ix)
        # Append gold heads tensor to the target heads list
        target_heads.append(torch.LongTensor(gold_heads))

    # Convert the target POS indices into a tensor
    target_pos_ixs = torch.LongTensor(target_pos_ixs)

    #########################################################
    # Calculate the scores with the model
    #########################################################

    # Calculate all the scores in a batch
    pred_pos_scores, pred_head_scores = zip(*tagger.forwards(inputs))

    #########################################################
    # Calculate the POS tagging-related loss
    #########################################################

    # Concatenate POS scores
    pred_pos_scores = torch.cat(pred_pos_scores)
    # Create a cross entropy object
    loss = nn.CrossEntropyLoss()
    # Calculate the POS tagging-related loss
    pos_loss = loss(pred_pos_scores, target_pos_ixs)

    #########################################################
    # Calculate the dependency parsing-related loss
    #########################################################

    # Store the dependency parsing loss in a variable
    dep_loss = 0.0
    # Calculate the loss for each sentence separately
    for pred_scores, targets in zip(pred_head_scores, target_heads):
        dep_loss += loss(pred_scores, targets)

    #########################################################
    # Return the total loss
    #########################################################

    # Return the sum of POS loss and dependency loss (you could also
    # use a weighted average, for instance, to indicate that either
    # dependency parsing or POS tagging is more important)
    return pos_loss + dep_loss


def total_loss(tagger: Tagger, data_set: Iterable[Sent]) -> TT:
    """Calculate the total cross entropy loss over the given dataset.

    The total loss is defined as a sum of the POS tagging-related loss and
    the dependency parsing-related loss.
    """

    #########################################################
    # Calculate the inputs and the target outputs
    #########################################################

    # Create a list for input sentences
    inputs = []          # type: List[Sequence[Word]]
    # Create two lists for target indices (POS tags and head indices)
    target_pos_ixs = []  # type: List[int]
    target_heads = []    # type: List[Head]

    # Loop over the dataset in order to determine the target indices
    for sent in data_set:
        # Extract the input words, gold POS tags, and gold dependency heads
        words = map(lambda tok: tok.word, sent)
        gold_tags = map(lambda tok: tok.upos, sent)
        gold_heads = map(lambda tok: tok.head, sent)
        # Append the new sentence to the inputs list
        inputs.append(list(words))
        # Determine the target POS tag indices
        for tag in gold_tags:
            # Determine the index corresponding to the gold POS tag
            ix = list(tagger.tagset).index(tag)
            # Append it to the target list
            target_pos_ixs.append(ix)
        # Extend the target heads list
        target_heads.extend(gold_heads)

    # Convert the target POS indices into a tensor
    target_pos_ixs = torch.LongTensor(target_pos_ixs)

    # Convert the target dependency indices into a tensor
    target_heads = torch.LongTensor(target_heads)

    #########################################################
    # Calculate the scores with the model
    #########################################################

    # Calculate all the scores in a batch
    pred_pos_scores, pred_head_scores = zip(*tagger.forwards(inputs))

    #########################################################
    # Calculate the POS tagging-related loss
    #########################################################

    # Concatenate POS scores
    pred_pos_scores = torch.cat(pred_pos_scores)
    # Create a cross entropy object
    loss = nn.CrossEntropyLoss()
    # Calculate the POS tagging-related loss
    pos_loss = loss(pred_pos_scores, target_pos_ixs)

    #########################################################
    # Calculate the dependency parsing-related loss
    #########################################################

    # Pad and concatenate dependency scores
    maxlen = max(scores.shape[1] for scores in pred_head_scores)
    pred_head_scores = torch.cat(list(
        # We use padding because the number of possible indices is different
        # for each sentence (in case of POS tagging this is easier, because
        # the number of POS tags is fixed); by using a very low padding
        # value (e.g., -10000.) we indicate that the model doesn't consider
        # them as valid heads
        F.pad(scores, (0, maxlen - len(scores)), value=-10000.)
        for scores in pred_head_scores
    ))

    # Calculate the dependency parsing-related loss
    dep_loss = loss(pred_head_scores, target_heads)

    #########################################################
    # Return the total loss
    #########################################################

    # Return the sum of POS loss and dependency loss (you could also
    # use a weighted average, for instance, to indicate that either
    # dependency parsing or POS tagging is more important)
    return pos_loss + dep_loss
