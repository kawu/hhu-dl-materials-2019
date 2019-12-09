from typing import Set, Dict, Iterator, Sequence

import torch
import random
import math

from core import TT, Name, Lang, DataSet
import names
from module import Module
from embedding import EmbeddingSum
from encoding import Encoding
from ffn import FFN
import utils
from utils import avg

# from optimizer import Optimizer
from torch.optim import Adam


# Note: we could also represent n-grams as tuples rather than strings.
# That would be more generic and allow, e.g., to use `None` to represent
# the beginning/ending of a string.
def ngrams(x: str, n: int) -> Iterator[str]:
    """Retrieve the sequence of n-grams in the given string.

    Arguments:
        n: size of n-grams
        x: input string

    >>> assert list(ngrams("abcd", 1)) == ["a", "b", "c", "d"]
    >>> assert list(ngrams("abcd", 2)) == ["ab", "bc", "cd"]
    """
    for i in range(len(x)-n+1):
        yield x[i:i+n]


class LangRec(Module):

    def __init__(self,
                 data_set: DataSet,
                 ngram_size: int,
                 emb_size: int,
                 hid_size: int):
        """Initialize the language recognition module.

        Args:
            data_set: the dataset from which the set of input symbols
                and output classes (languages) can be extracted
            ngram_size: size of n-gram features (e.g., use 1 for unigrams,
                2 for bigrams, etc.)
            emb_size: size of the character embedding vectors
            hid_size: size of the hidden layer of the FFN use for scoring
        """
        # Keep the size of the ngrams
        self.ngram_size = ngram_size
        # Calculate the embedding alphabet and create the embedding sub-module
        feat_set = self.alphabet(data_set)
        self.register("emb", EmbeddingSum(feat_set, emb_size))
        # Encoding (mapping between langs and ints)
        lang_set = set(lang for (_, lang) in data_set)
        self.enc = Encoding(lang_set)
        # Scoring FFN sub-module
        self.register("ffn",
                      FFN(idim=emb_size,
                          hdim=hid_size,
                          odim=len(lang_set))
                      )
        # Additional check to verify that all the registered
        # parameters actually require gradients.
        assert all([param.requires_grad is True for param in self.params()])

    def preprocess(self, name: Name) -> Name:
        """Name preprocessing."""
        # Currently no preprocessing, but we could think of something
        # in the future.
        return name

    def features(self, name: Name) -> Iterator[str]:
        """Retrieve the list of features in the given name."""
        return ngrams(self.preprocess(name), self.ngram_size)

    def alphabet(self, data_set: DataSet) -> Set[str]:
        """Retrieve the embedding alphabet from the dataset.

        Retrieve the set of all features that we want to embed from
        the given dataset.
        """
        return set(feat
                   for (name, _) in data_set
                   for feat in self.features(name)
                   )

    def encode(self, lang: Lang) -> int:
        """Encode the given language as an integer."""
        return self.enc.encode(lang)

    def forward(self, names: Iterator[Name]) -> TT:
        """The forward calculation of the name's language recognition model.

        Args:
            names: a sequence of person names; calculating the scores for
                several names at the same time is faster thanks to better
                parallelization

        Returns:
            score matrix in which each row corresponds to a single name, with
            its individual elements corresponding to the scores of different
            languages
        """
        # TODO EX2 (a): the following lines need to be adapted to the EmbeddingSum,
        # which processes features in groups.  You will also need to make
        # trivial modifications in the code in two or three other places
        # (imports, initialization).
        # TODO EX2 (b): you can further try to modify the EmbeddingSum class so
        # that it works over batches of feature groups.
        embeddings = [
            # [self.emb.forward(feat) for feat in self.features(name)]
            self.emb.forward(self.features(name))
            for name in names
        ]
        # cbow = utils.from_rows(map(sum, embeddings))
        cbow = utils.stack(embeddings)
        scores = self.ffn.forward(cbow)
        return scores

    def classify(self, name: Name) -> Dict[Lang, float]:
        """Classify the given person name.

        Args:
            name: person name, sequence of characters

        Returns:
            the mapping from languages to their probabilities
            for the given name.
        """
        # We don't want Pytorch to calculate the gradients
        with torch.no_grad():
            # The vector of scores for the given name
            scores = self.forward([name])[0]
            # We map the vector of scores to the vector of probabilities.
            probs = torch.softmax(scores, dim=0)
            # Result dictionary
            res = {}
            # `ix` should be an index in the scores vector
            for ix in range(len(probs)):
                lang = self.enc.decode(ix)
                res[lang] = probs[ix]
            return res

    def classify_one(self, name: Name) -> Lang:
        """A simplified version of `classify` which returns the
        language with the highest score."""
        prob_map = self.classify(name)
        preds = sorted(prob_map.items(), key=lambda pair: pair[1])
        (name, _prob) = preds[-1]
        return name


def single_loss(output: TT, target: int) -> TT:
    """Calculate the loss between the predicted scores and
    the target class index.

    Args:
        output: vector of scores predicated by the model
        target: the index of the target class
    """
    # Additional checks
    assert len(output.shape) == 1          # output is a vector
    assert 0 <= target < output.shape[0]   # target is not out of range
    # Return the cross entropy between the output score vector and
    # the target ID.
    return torch.nn.CrossEntropyLoss()(
        output.view(1, -1),         # Don't worry about the view method for now
        torch.tensor([target])
    )
    # It would be more intuitive to calculate the predicted distribution
    # before calculating the cross-entropy.  This is not done for numerical
    # reasons (the backpropagation algo wouldn't work).


def batch_loss(outputs: TT, targets: Sequence[int]) -> TT:
    """Calculate the loss between the predicted scores and
    the target class index.

    Args:
        outputs: matrix of scores predicated by the model
        target: the index of the target class
    """
    # Additional checks
    assert len(outputs.shape) == 2
    assert all(
        0 <= target < outputs.shape[1]
        for target in targets
    )
    # Return the cross entropy between the output score matrix and
    # the target IDs.
    return torch.nn.CrossEntropyLoss()(
        outputs, torch.tensor(targets)
    )


def total_loss(data_set: DataSet, lang_rec: LangRec):
    """Calculate the total loss of the model on the given dataset."""
    # Calculate the target language identifiers over the entire dataset
    target_lang_ids = [
        lang_rec.encode(lang)
        for (_, lang) in data_set
    ]
    # Determine the names to run our network on
    names = (name for (name, _) in data_set)
    # Predict the scores for all the names in parallel
    predicted_scores = lang_rec.forward(names)
    # Calculate the loss
    return batch_loss(predicted_scores, target_lang_ids)


def print_predictions(lang_rec: LangRec, data_set: DataSet, show_max=5):
    """Print predictions of the model on the given data_set.

    Keyword arguments:
        show_max: the maximum number of languages to show
    """
    for (name, lang) in data_set:
        pred = sorted(lang_rec.classify(name).items(),
                      key=lambda pair: pair[1], reverse=True)
        print("{name}, {targ} => {pred}".format(
            name=name, pred=pred[:show_max], targ=lang))


def accuracy(lang_rec: LangRec, data_set: DataSet) -> float:
    """Calculate the accuracy of the model on the given dataset.

    The accuracy is defined as the percentage of the names in the data_set
    for which the lang_rec model predicts the correct language.
    """
    k, n = 0, 0
    for (name, target_lang) in data_set:
        pred_lang = lang_rec.classify_one(name)
        if target_lang == pred_lang:
            k += 1
        n += 1
    return k / n


def train(
        train_set: DataSet,
        dev_set: DataSet,
        lang_rec: LangRec,
        learning_rate=1e-3,
        report_rate=10,
        epoch_num=50,
        mini_batch_size=50
):
    """Train the model on the given dataset w.r.t. the total_loss function.
    The model is updated in-place.

    Arguments:
        train_set: the dataset to train on
        dev_set: the development dataset
        lang_rec: the language recognition model
        learning_rate: hyper-parameter of the SGD method
        report_rate: how often to report the loss on the training set
        epoch_num: the number of SGD epochs
        mini_batch_size: size of the mini-batch
    """
    # # Create our optimizer
    # optim = Optimizer(lang_rec.params(),
    #                   learning_rate=learning_rate)

    # Use the Adam optimizer provided by PyTorch
    optim = Adam(lang_rec.params(),
                 lr=learning_rate)

    # How many updates to perform in an epoch
    iter_in_epoch = math.ceil(len(train_set) / mini_batch_size)

    # Perform gradient-descent in a loop
    for t in range(epoch_num):
        # For each epoch, perform a number of mini-batch updates
        for _ in range(iter_in_epoch):
            # Determine the mini-batch
            mini_batch = random.sample(train_set, mini_batch_size)
            # Calculate the total loss
            loss = total_loss(mini_batch, lang_rec)
            # Calculate the gradients of all parameters
            loss.backward()
            # Optimizer step
            optim.step()
            # Zero-out the gradients
            optim.zero_grad()

        # Reporting
        if (t+1) % report_rate == 0:
            with torch.no_grad():
                msg = ("@ {k}: "
                       "loss(train)={tl}, acc(train)={ta}, "
                       "loss(dev)={dl}, acc(dev)={da}")
                print(msg.format(
                    k=t+1,
                    tl=round(total_loss(train_set, lang_rec).item(), 3),
                    ta=round(accuracy(lang_rec, train_set), 3),
                    dl=round(total_loss(dev_set, lang_rec).item(), 3),
                    da=round(accuracy(lang_rec, dev_set), 3))
                )


# In the main function, the grid search method is used to help in determining
# the adequate values of the hyperparameters.
def main():

    # Training and development dataset (you can find those on the webpage:
    # https://user.phil.hhu.de/~waszczuk/teaching/hhu-dl-wi19/names/split.zip
    # train_set = names.load_data("split/dev80.csv")
    # dev_set = names.load_data("split/dev20.csv")
    train_set = names.load_data("split/train.csv")
    dev_set = names.load_data("split/dev.csv")
    print("Train size:", len(train_set))
    print("Dev size:", len(dev_set))

    # Size of n-grams
    ng_size = 2
    # Numer of epochs (one training)
    epoch_num = 10
    # Reporting rate (freq?)
    rep_rate = epoch_num+1  # no reporting
    # Initial learning rate
    init_lr = 0.01
    # Mini-batch size
    mb_size = 256

    # Number of trials per each hyper-param combination to get (more)
    # reliable results
    tries = 5

    # Hyper-param values to consider (a rather coarse grid, you can try
    # something more fine-grained, but that would of course increase
    # the computation time)
    emb_size_list = [10, 50, 100]
    hid_size_list = [10, 50, 100]

    for emb_size in emb_size_list:
        for hid_size in hid_size_list:
            print("# emb_size={0}, hid_size={1}".format(
                emb_size, hid_size))
            train_loss = []
            train_acc = []
            dev_loss = []
            dev_acc = []
            for _ in range(tries):
                lang_rec = LangRec(
                    train_set,
                    emb_size=emb_size,
                    hid_size=hid_size,
                    ngram_size=ng_size
                )

                # Training
                train(train_set, dev_set, lang_rec, epoch_num=epoch_num,
                      learning_rate=init_lr, report_rate=rep_rate,
                      mini_batch_size=mb_size)

                # Loss and accuracy
                with torch.no_grad():
                    train_loss.append(total_loss(train_set, lang_rec).item())
                    train_acc.append(accuracy(lang_rec, train_set))
                    dev_loss.append(total_loss(dev_set, lang_rec).item())
                    dev_acc.append(accuracy(lang_rec, dev_set))
                    msg = ("loss(train)={tl}, acc(train)={ta}, "
                           "loss(dev)={dl}, acc(dev)={da}")
                    print("@", msg.format(
                        tl=round(train_loss[-1], 3),
                        ta=round(train_acc[-1], 3),
                        dl=round(dev_loss[-1], 3),
                        da=round(dev_acc[-1], 3))
                    )

            # Print scores averaged over the trial runs
            msg = ("loss(train)={tl}, acc(train)={ta}, "
                   "loss(dev)={dl}, acc(dev)={da}")
            print("# AVG:", msg.format(
                tl=round(avg(train_loss), 3),
                ta=round(avg(train_acc), 3),
                dl=round(avg(dev_loss), 3),
                da=round(avg(dev_acc), 3))
            )
