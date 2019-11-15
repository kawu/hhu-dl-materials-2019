from typing import Sequence
import torch

from core import TT


class Optimizer:

    """Optimizer encapsulates the functionality related to
    updating parameters based on their gradients.
    """

    def __init__(self,
                 params: Sequence[TT],
                 learning_rate: float = 1e-3):
        self.params = params
        self.learning_rate = learning_rate

    def step(self):
        """Perform a single step of optimization."""
        with torch.no_grad():
            # Update the gradient for each parameter of the model.  Since
            # the model is an instance of the Module class, we can access
            # all its parameters using the params method.
            for param in self.params:
                if param.grad is not None:
                    param -= self.learning_rate * param.grad
                    param.grad.zero_()
