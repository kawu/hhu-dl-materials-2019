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
            # not in gradient descent!  In SGD, not all parameters have
            # to be involved each time we calculate the gradients.
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
entire training dataset (which before was problematic with the simple GD).

<!---
**Note**: The two techniques are not always distinguished, perhaps because
using the ,,history of gradients'' is especially beneficial for SGD.
Nevertheless, they are independent in that Momentum or Adam could be in
principle used with standard GD and, vice-versa, SGD doesn't require Momentum
or Adam.
-->

# SGD


# Optimization methods


# Refactoring

* Optimizer
* Training method in a separate module?


[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS "Limited-memory BFGS"
