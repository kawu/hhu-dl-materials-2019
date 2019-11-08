# Gradient descent

Currently, our training method is based on [gradient descent
(GD)](https://en.wikipedia.org/wiki/Gradient_descent) and its implementation
looks like this:
```python
# Perform gradient-descent in a loop
for t in range(epoch_num):
    # Calculate the total loss
    loss = total_loss(data_set, model)
    # Calculate the gradients of all parameters
    loss.backward()
    # Update the parameters
    with torch.no_grad():
        # Update each parameter of the model based on its gradient
        for param in lang_rec.params():
            # TODO: In fact, `param.grad` should not be None, especially
            # not in batch gradient descent!  In SGD, not all parameters 
            # have to be involved each time we calculate the gradients.
            # We could set them to 0 manually, though.
            if param.grad is not None:
                param -= learning_rate * param.grad
                param.grad.zero_()
```

This strategy, however, is typically not effective enough to training
large-scale neural networks (nor to solve other non-trivial optimization
problems).

There are (at least) two techniques for improving GD.  The first one involves
better utilization of the gradients.  In GD, each step is based solely on the
current gradient, regardless of the history of the previously calculated
gradients.  This is something that [optimization
methods](#optimization-methods) such as Momentum, Adam, or [L-BFGS](l-bfgs)
improve upon.  The second technique is to introduce some randomization in the
training procedure, which leads to [stochastic gradient descent (SGD)](#SGD).

Today, we discuss both these techniques and use them to improve our
implementation.  In particular, SGD will allow us to train the model on the
entire training dataset (which would be problematic with simple GD).

<!---
**Note**: The two techniques are not always distinguished, perhaps because
using the ,,history of gradients'' is especially beneficial for SGD.
Nevertheless, they are independent in that Momentum or Adam could be in
principle used with standard GD and, vice-versa, SGD doesn't require Momentum
or Adam.
-->

# SGD

In standard gradient descent, also called *batch gradient descent*, we
calculate in each step the gradient over the entire training dataset.  In
*stochastic gradient descent*, in each step we randomly sample a subset of the
training dataset and perform the calculation of the gradient w.r.t. this random
sample.  This sample is typically called a *mini-batch*.  In Python:
```python
iter_num = 1000         # number of SGD iterations
mini_batch_size = 50    # mini-batch size
for t in range(iter_num):
    # Sample the mini_batch
    mini_batch = random.sample(mini_batch_size, data_set)
    # Calculate the loss over the mini-batch
    loss = total_loss(mini_batch, model)
    # Calculate the gradients
    loss.backward()
    # Update the gradients
    ...
```
The advantage of SGD over batch GD is that the calculation of the gradients is
much more frequent.  Thanks to that, the procedure is much faster to converge.
On the other hand, we sample our mini-batches, hence the variance between the
gradients calculated in the subsequent iterations will be higher than with
batch GD.  To alleviate the issue:
* A progressively decreasing learning rate can be used
* The size of the mini-batch can be increased

The choice of the size of the mini-batch is thus a trade-off between:
* The frequency with which gradients are calculated (the higher the better,
  since convergence is faster)
* The variance between the subsequently calculated gradients (the lower the
  better, because otherwise convergence gets unstable, i.e., ,,jumps'' a lot)
Additionally, larger mini-batch size enables better parallelization (one can
calculate the gradient over several training examples in parallel, for
instance).

Finally, a form of SGD which is typically used is slightly different from the
code presented above.  Namely, to make sure that each element from the training
dataset is ,,visited'' a roughly equal number of times:
```python
epoch_num = 10          # num. of iterations over the training dataset
mini_batch_size = 50    # mini-batch size
mini_batch_step = 10    # TODO
for t in range(epoch_num):
    # First shuffle the dataset to introduce randomization
    random.shuffle(data_set)
    # Dataset size
    n = len(data_set)
    # Iterate over mini_batches
    for k in range(0, len(data_set), mini_batch_step):
        # Create the mini-batch starting from position k
        mini_batch = data_set[k:k+mini_batch_size] + ...
        # Calculate the loss over the mini-batch
        loss = total_loss(mini_batch, model)
        # Calculate the gradients
        loss.backward()
        # Update the gradients
        ...
```

# Optimization methods

TODO


# Refactoring

Refactoring ideas:
* Optimizer
* Training method in a separate module


[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS "Limited-memory BFGS"
