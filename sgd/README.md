# Gradient descent

Currently, our training method is based on [gradient descent
(GD)](https://en.wikipedia.org/wiki/Gradient_descent) and its implementation
looks like this:
```python
# Perform gradient-descent in a loop
for t in range(epoch_num):
    # Calculate the loss, which we want to minimize
    loss = total_loss(data_set, model)
    # Calculate the gradients of all the parameters via backpropagation
    loss.backward()
    # Update the parameters
    with torch.no_grad():
        # Update each parameter of the model in the opposite direction
        # of its gradient
        for param in model.params():
            if param.grad is not None:
                param -= learning_rate * param.grad
                param.grad.zero_()
```

This strategy, however, is typically not effective enough to training
large-scale neural networks (nor to solve other non-trivial optimization
problems).

There are (at least) two ways to improve GD.  The first one involves
better utilization of the gradients.  In GD, each step is based solely on the
current gradient, regardless of the history of the previously calculated
gradients.  This is something that [optimization
methods](#optimization-methods) such as Momentum, Adam, or [L-BFGS][l-bfgs]
improve upon.  The second technique is to introduce randomization in the
training procedure, which leads to [stochastic gradient descent (SGD)](#SGD).

Today, we discuss both these techniques and use them to improve our
implementation.  In particular, SGD will allow us to train the model on the
entire training dataset (which would be problematic with simple GD).
<!--- Some of the things we implement today are already available in PyTorch,
but we will
-->

<!---
**Note**: The two techniques are not always distinguished, perhaps because
using the ,,history of gradients'' is especially beneficial for SGD.
Nevertheless, they are independent in that Momentum or Adam could be in
principle used with standard GD and, vice-versa, SGD doesn't require Momentum
or Adam.
-->

**Exercise**: Encapsulate the parameter updating aspect of GD in a new
`Optimizer` class (as in PyTorch).  For initialization, the `Optimizer` should
take the list of parameters to update in each step of GD.  It should also
provide one method, `step()` (with no arguments), which updates the parameters
w.r.t to their gradients.  Currently this functionality is implemented inline
in the training method:
```python
# Update the parameters
with torch.no_grad():
    # Update each parameter of the model in the opposite direction
    # of its gradient
    for param in model.params():
        if param.grad is not None:
            param -= learning_rate * param.grad
            param.grad.zero_()
```

**Note**: Mathematically speaking, there's only one gradient calculated in each
step.  However, we often speak of gradients (plural), because the
(mathematical) gradient is structured and separated into several (phisical)
gradient vectors, one per parameter tensor.

# SGD

In standard gradient descent, also called *batch gradient descent*, we
calculate (in each step) the gradient over the entire training dataset.  In
*stochastic gradient descent*, in each step we randomly sample a subset of the
training dataset and perform the calculation of the gradient w.r.t. this random
sample.  This sample is typically called a *mini-batch*.  In Python:
```python
iter_num = 1000         # number of SGD iterations
mini_batch_size = 50    # mini-batch size
for t in range(iter_num):
    # Sample the mini_batch
    mini_batch = random.sample(data_set, mini_batch_size)
    # Calculate the loss over the mini-batch
    loss = total_loss(mini_batch, model)
    # Calculate the gradients
    loss.backward()
    # Update the parameters
    ...
```

The advantage of SGD over batch GD is that the calculation of the gradients is
much more frequent.  Thanks to that, the procedure is faster to converge.  On
the other hand, we sample our mini-batches, hence the variance between the
gradients calculated in the subsequent iterations will be higher than with
batch GD.  To alleviate the issue:
* A progressively decreasing learning rate can be used
* The size of the mini-batch can be increased

The choice of the size of the mini-batch is thus a trade-off between:
* The frequency with which gradients are calculated (the higher the better,
  since convergence is faster)
* The variance between the subsequently calculated gradients (the lower the
  better, because otherwise convergence gets unstable)

Additionally, a larger mini-batch size enables better parallelization (one can
calculate the gradients over several training examples in parallel, for
instance).

Finally, a form of SGD which is typically used is slightly different from the
code presented above.  Namely, to make sure that each element from the training
dataset is ,,visited'' a roughly equal number of times:
```python
epoch_num = 10          # num. of iterations over the training dataset
mini_batch_size = 50    # size of the mini-batches
mini_batch_step = 25    # to speed up training (somewhat unorthodox)
for t in range(epoch_num):
    # First shuffle the dataset to introduce randomization (note: data_set
    # should be a shallow copy of the original training set)
    random.shuffle(data_set)
    # Dataset size
    n = len(data_set)
    # Iterate over mini_batches
    for k in range(0, len(data_set), mini_batch_step):
        # Create the mini-batch starting from position k
        mini_batch = data_set[k:k+mini_batch_size]
        # Just make sure that the mini_batch is not empty
        assert len(mini_batch) > 0
        # Calculate the loss over the mini-batch
        loss = total_loss(mini_batch, model)
        # Calculate the gradients
        loss.backward()
        # Update the parameters
        ...
```

**Exercise**: Replace the implementation of gradient descent in our language
recognition code with stochastic gradient descent.  You can create a new class
responsible for splitting the dataset into mini-batches. Verify that this still
allows to train the model on `dev80.csv`.  It should be also faster.

**Exercise**: Is the size of each mini-batch equal to `mini_batch_size` in the
code above?  If not, what can we do to improve this?

**Note**: The size of the mini-batch may have an impact on the size of the
gradients.  Thus, you may need to adapt the learning rate depending on the
mini-batch size.

<!---
**Note**: SGD relies on the total loss being defined as the sum of the losses
for the individual dataset elements.  If you break this assumption, SGD won't
work properly.
-->

# Optimization methods

**Note**.  A broad, rather informal overview of different gradient optimization
algorithms used in deep learning can be found [here][overview-sgd].  We will
focus on two algorithms today (Momentum and Adam).
<!---
You can in particular have a look at the [visualization of the SGD
algorithms][overview-visualization].
-->
<!--- In practice, the *Adam* algorithm is typically -->

In standard (S)GD, in each iteration of the optimization procedure, the
gradient is computed with respect to the current (mini-)batch. The parameters
are then updated in the opposite direction of the gradient (scaled by the
`learning_rate`):
```python
...
for param in model.params():
    param -= learning_rate * param.grad
    param.grad.zero_()
...
```

The standard SGD is known to have trouble dealing with complex models that
arise in deep learning, which lead to complex objective function (objective
functions are the ones that we want to minimize).  In particular, standard SGD
is poor in dealing with ravines (see [momentum][overview-momentum]), [saddle
points][saddle], or plateaus (flat regions in the parameter search space).

SGD completely ignores the previously calculated gradients, which is one of the
reasons behind its poor performance.  This is also the issue that the
optimization algorithms popular in deep learning try to resolve.  We consider
two basic techniques (*momentum* and *adaptive learning rates*) below.

### Momentum

A pretty good explanation of momentum can be found in the [corresponding
section of the overview][overview-momentum] mentioned above.

SGD with momentum is pretty good, but often not enough to successfully train a
neural network.  On the other hand, it is easy to understand and
straightforward to implement.

**Exercise** (advanced): Implement SGD with momentum based on the [description
in the overview][overview-momentum].  In particular, replace the `Optimizer`
class with a version that keeps track of the momentums.  It should have the
same interface as `Optimizer` (i.e., `__init__` should take the list of
parameter tensors and `step()` should take no arguments).

### Adaptive learning rates

Adaptive learning rates is a technique that involves setting the learning rates
for the individual parameters of the neural network separately.  
<!---
This allows, for instance, to set a low learning rate for frequently activated
parameters and high learning rates for rarely activated parameters.
-->

For example, imagine that a particular character (e.g., `ü`) occurs a very
small number of times in our language recognition training dataset, in contrast
to e.g. `a`.  Each time we calculate the loss over a mini-batch with `ü` , the
embedding parameters corresponding to `ü` get updated.  Similarly, each time we
calculate the loss over a mini-batch with `a` , the embedding parameters
corresponding to `a` get updated.   However, the latter occur much more often
than the former, which means that we will update the `a`-related parameters 
many times, and the `ü`-related parameters only a few times, in one training
epoch.  Therefore, we would like the learning rates for the `a`-related
parameters to be smaller than the learning rates for the `ü`-related
parameters.

### Adam

An optimization algorithm that uses both momentum and adaptive learning rates
is *Adam*.  You can find more details about Adam in the [corresponding section
of the overview][overview-adam].  Adam seems to be the default choice as far as
training neural networks is concerned.

**Note**.  You may encounter opinions such as ,,SGD with momentum should be
used for task A, Adam for task B'', etc.  Take such statements with a grain of
salt.  As long as you are able to train the network on the training dataset,
and you obtain good scores (e.g., accuracy) on train (which means that the
network does not [underfit][underfitting]), you shouldn't worry about the
particular optimization algorithm you use.  It is only when you are not able to
fit your model to the training dataset that you can consider changing the
optimization method.

**Exercise**: Use the `Optimizer` from PyTorch, set it up to use `Adam`, and
substitue it for the optimizer currently used in the code.  Verify that
training still works, i.e., that you can obtain a similar (or lower) level of
loss over the training set at the end of the training process.

<!---
# Exercises

* Implement standard SGD with mini-batches.  Make sure that you are still able
  to train the model on `dev80.csv` and that the resulting loss is not worse
  than with standard GD.  Then, see if you can train the model over the entire
  *train.csv* dataset.
* Replace the basic optimizer with Adam from PyTorch.
* Simple drop-out technique: just randomly dropping some of the features on
  input.
* Optional Currently, our model will assign different scores to *jones*,
  *Jones*, and *JONES*.  Propose a modification of the model which will
  guarantee that these three versions lead to the same score distribution.
-->



<!---
# Refactoring

Refactoring ideas:
* Optimizer
* Training method in a separate module
-->


[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS "Limited-memory BFGS"
[overview-sgd]: https://ruder.io/optimizing-gradient-descent/ "Overview of GD algorithms"
[overview-visualization]: https://ruder.io/optimizing-gradient-descent/index.html#visualizationofalgorithms "Visualization of SGD algorithms"
[overview-momentum]: https://ruder.io/optimizing-gradient-descent/index.html#momentum "SGD with Momentum"
[saddle]: https://en.wikipedia.org/wiki/Saddle_point "Saddle point"
[underfitting]: https://en.wikipedia.org/wiki/Overfitting#Underfitting
