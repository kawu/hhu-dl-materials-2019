import torch.nn as nn
# import torch.nn.functional as F

from neural.types import TT


class MLP(nn.Module):
    """Multi-layered perceptron (also called feed-forwards network)"""

    def __init__(self, idim: int, hdim: int, odim: int):
        """Create a feed-forward network.

        Args:
            idim: size of the input vector
            hdim: size of the hidden vector
            odim: size of the output vector
        """
        super(MLP, self).__init__()
        self.L1 = nn.Linear(idim, hdim)
        self.L2 = nn.Linear(hdim, odim)
        self.activ = nn.LeakyReLU(inplace=True)

    def forward(self, X: TT) -> TT:
        # Explicitely check that the dimensions match
        assert X.shape[-1] == self.L1.in_features
        # Calculate the hidden layer and apply activation
        H = self.L1.forward(X)
        self.activ(H)
        # H = F.leaky_relu(self.L1.forward(X))
        return self.L2.forward(H)
