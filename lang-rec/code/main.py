from typing import Dict

import torch

from core import TT, Name, Lang, DataSet
import names
from module import Module

from embedding import Embedding
from encoding import Encoding
from ffn import FFN

from optimizer import Optimizer


def char_set_in(data_set: DataSet) -> set:
    """Retrieve the set of chars in the dataset."""
    name_list = [name for (name, lang) in data_set]
    char_matrix = [list(name) for name in name_list]
    def flatten(l): return [item for sublist in l for item in sublist]
    char_set = set(flatten(char_matrix))
    return char_set


def char_set_in_alt(data_set: DataSet):
    """Alternative version of char_set_in."""
    return set(
        char
        for (name, lang) in data_set
        for char in name
    )


class LangRec(Module):

    # TODO: Implement this method.
    def __init__(self,
                 data_set: DataSet,
                 emb_size: int,
                 hid_size: int):
        """Initialize the language recognition module.

        Args:
            data_set: the dataset from which the set of input symbols
                and output classes (languages) can be extracted
            emb_size: size of the character embedding vectors
            hid_size: size of the hidden layer of the FFN use for scoring
        """
        char_set = char_set_in(data_set)
        # Embedding
        self.register("emb", Embedding(char_set, emb_size))
        lang_set = set(
            lang for (name, lang) in data_set
        )
        lang_num = len(lang_set)
        # Encoding (mapping between langs and ints)
        self.enc = Encoding(lang_set)
        # FFN
        self.register("ffn",
                      FFN(idim=emb_size, hdim=hid_size, odim=lang_num))

    # TODO: Implement this method.
    def encode(self, lang: Lang) -> int:
        """Encode the given language as an integer."""
        return self.enc.encode(lang)

    # TODO: Implement this method.
    def forward(self, name: Name) -> TT:
        """The forward calculation of the name's language recognition model.

        Args:
            name: a person name

        Returns:
            score vector corresponding to the name, with its individual
            elements corresponding to the scores of different languages
        """
        embeddings = [self.emb.forward(char) for char in name]
        cbow = sum(embeddings)
        scores = self.ffn.forward(cbow)
        return scores

    # TODO: Implement this method.

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
            scores = self.forward(name)
            # We map the vector of scores to the vector of probabilities.
            probs = torch.softmax(scores, dim=0)
            # Result dictionary
            res = {}
            # `ix` should be an index in the scores vector
            for ix in range(len(probs)):
                lang = self.enc.decode(ix)
                res[lang] = probs[ix]
            return res


def single_loss(output: TT, target: int) -> TT:
    """Calculate the loss between the predicted scores and
    the target class index.

    Args:
        output: vector of scores predicated by the model
        target: the index of the target class
    """
    # Additional checks
    assert len(output.shape) == 1          # output is a vector
    assert 0 <= target <= output.shape[0]  # target is not out of range
    # Return the cross entropy between the output score vector and
    # the target ID.
    return torch.nn.CrossEntropyLoss()(
        output.view(1, -1),         # Don't worry about the view method for now
        torch.tensor([target])
    )
    # It would be more intuitive to calculate the predicted distribution
    # before calculating the cross-entropy.  This is not done for numerical
    # reasons (the backpropagation algo wouldn't work).


def total_loss(data_set: DataSet, lang_rec: LangRec):
    """Calculate the total loss of the model on the given dataset."""
    # The variable to keep the loss
    loss = torch.tensor(0.0)
    for (name, lang) in data_set:
        # First calculate the ID of the target language
        # lang_id = one_hot_inv(lang_rec.enc.encode(lang))
        lang_id = lang_rec.encode(lang)
        # Predict the scores
        scores = lang_rec.forward(name)
        # Update the loss
        loss += single_loss(scores, lang_id)
    return loss


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


# TODO: Implement this function.
def accuracy(lang_rec: LangRec, data_set: DataSet) -> float:
    """Calculate the accuracy of the model on the given dataset.

    The accuracy is defined as the percentage of the names in the data_set
    for which the lang_rec model predicts the correct language.
    """
    pass


def train(
        data_set: DataSet,
        lang_rec: LangRec,
        learning_rate=1e-3,
        report_rate=10,
        epoch_num=50
):
    """Train the model on the given dataset w.r.t. the total_loss function.
    The model is updated in-place.

    Args:
        data_set: the dataset to train on
        lang_rec: the language recognition model
        learning_rate: hyper-parameter of the gradient descent method
        report_rate: how often to report the loss on the training set
        epoch_num: the number of epochs the training procedure
    """
    # Create our optimizer
    optim = Optimizer(lang_rec.params(),
                      learning_rate=learning_rate)

    # Perform gradient-descent in a loop
    for t in range(epoch_num):

        # Calculate the total loss
        loss = total_loss(data_set, lang_rec)
        # Calculate the gradients of all parameters
        loss.backward()
        # Optimizer step
        optim.step()

        # with torch.no_grad():
        #     # Update the gradient for each parameter of the model.  Since
        #     # the model is an instance of the Module class, we can access
        #     # all its parameters using the params method.
        #     for param in lang_rec.params():
        #         if param.grad is not None:
        #             param -= learning_rate * param.grad
        #             param.grad.zero_()

        # Reporting
        if (t+1) % report_rate == 0:
            with torch.no_grad():
                # TODO: you can also report the accurracy on the dev set
                # (once the accuracy function is implemented)
                loss = total_loss(data_set, lang_rec)
                print(t+1, loss.item())


# # The main script of tha application, put in the `main` function
# # so you can `run main` from IPython before filling in all the TODOs.
# def main():

# Training and development dataset
train_set = names.load_data("split/dev80.csv")
dev_set = names.load_data("split/dev20.csv")
print("Train size:", len(train_set))
print("Dev size:", len(dev_set))

# Language recognition model
lang_rec = LangRec(
    train_set,
    # TODO: How to choose the embedding size?
    emb_size=10,
    hid_size=100
)

# TODO: fill in the rest of the script here
