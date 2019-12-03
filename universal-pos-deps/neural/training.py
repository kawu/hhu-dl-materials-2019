from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from neural.types import TT


def batch_loader(data_set: IterableDataset,
                 batch_size: bool,
                 shuffle=True) -> DataLoader:
    """Create a batch data loader from the given data set.

    Using PyTorch DataSets and DataLoaders is especially useful when working
    with large datasets, which cannot be stored in the computer memory (RAM)
    all at once.

    Let's create a small dataset of numbers:
    >>> data_set = range(5)
    >>> for elem in data_set:
    ...     print(elem)
    0
    1
    2
    3
    4

    The DataLoader returned by the batch_loader function allows to
    process the dataset in batches.  For example, in batches of
    2 elements:
    >>> bl = batch_loader(data_set, batch_size=2, shuffle=False)
    >>> for batch in bl:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4]

    The last batch is of size 1 because the dataset has 5 elements in total.
    You can iterate over the dataset in batches over again:
    >>> for batch in bl:
    ...     print(batch)
    [0, 1]
    [2, 3]
    [4]

    For the sake of training of a PyTorch model, it may be better to shuffle
    the elements each time the stream of batches is created.
    To this end, use the `shuffle=True` option.
    >>> bl = batch_loader(data_set, batch_size=2, shuffle=True)

    DataLoader "visits" each element of the dataset once.
    >>> sum(len(batch) for batch in bl) == len(data_set)
    True
    >>> set(x for batch in bl for x in batch) == set(data_set)
    True
    """
    return DataLoader(
        data_set,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        shuffle=shuffle
    )


def train(
        model: nn.Module,
        train_set: IterableDataset,
        dev_set: Optional[IterableDataset],
        total_loss: Callable[[nn.Module, IterableDataset], TT],
        accuracy: Callable[[nn.Module, IterableDataset], float],
        batch_size=32,
        learning_rate=1e-3,
        report_rate=10,
        epoch_num=50
):
    """Train the model on the given dataset w.r.t. the total_loss function.
    The model parameters are updated in-place.

    Args:
        model: the neural model to be trained
        train_set: the dataset to train on
        dev_set: the development dataset (can be None)
        total_loss: the objective function we want to minimize;
            note that this function must support backpropagation!
        accuracy: accuracy of the model over the given dataset
        batch_size: size of the SGD batches
        learning_rate: hyper-parameter of the SGD method
        report_rate: how often to report the loss/accuracy on train/dev
        epoch_num: the number of epochs of the training procedure
    """
    # Choose Adam for optimization
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    # Create batched loader
    batches = batch_loader(
        train_set, batch_size=batch_size, shuffle=True)

    # Perform SGD in a loop
    for t in range(epoch_num):

        # We use a PyTorch DataLoader to provide a stream of
        # dataset element batches
        for batch in batches:
            loss = total_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting (every `report_rate` epochs)
        if (t+1) % report_rate == 0:
            with torch.no_grad():
                train_loss = total_loss(model, train_set).item()
                train_acc = accuracy(model, train_set)
                if dev_set:
                    dev_acc = accuracy(model, dev_set)
                else:
                    dev_acc = 0.0
                msg = ("@{k}: "
                       "loss(train)={tl}, acc(train)={ta}, "
                       "acc(dev)={da}")
                print(msg.format(
                    k=t+1,
                    tl=round(train_loss, 3),
                    ta=round(train_acc, 3),
                    da=round(dev_acc, 3))
                )
