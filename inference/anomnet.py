"""This is a pared-down version of the AnomNet model from the original code,
where only the 1D convolutional layers and fully connected layers are retained.
https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_NN_utils.py
"""

from torch import nn
import torch.nn.functional as F
from torch import Tensor


class Flatten(nn.Module):
    """
    Module to flatten the input tensor except for the batch dimension.

    Methods
    -------
    forward(input: Tensor) -> Tensor
        Flattens the input tensor to shape (batch_size, -1).
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten the input tensor except for the batch dimension.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape (batch_size, -1).
        """
        return input.view(input.size(0), -1)


class AnomNet(nn.Module):
    """
    Anomaly detection neural network with 1D convolutional and fully connected layers.

    Parameters
    ----------
    init_filt : int, optional
        Number of input channels for the first convolutional layer (default is 5).
    n_filt : int, optional
        Number of output channels for convolutional layers (default is 5).
    kernel_size : int, optional
        Size of the convolutional kernel (default is 10).
    padding : int, optional
        Padding for convolutional layers (default is 0).
    stride : int, optional
        Stride for convolutional layers (default is 1).
    dropout : float, optional
        Dropout probability (default is 0).
    n_feat : int, optional
        Number of input features for the first fully connected layer (default is 500).
    init_neur : int, optional
        Number of input neurons for the second fully connected layer (default is 100).
    n_neur : int, optional
        Number of neurons in the hidden fully connected layers (default is 10).
    n_out : int, optional
        Number of output neurons (default is 1).

    Methods
    -------
    forward(x: Tensor) -> Tensor
        Forward pass of the network.
    """

    def __init__(
        self,
        init_filt: int = 5,
        n_filt: int = 5,
        kernel_size: int = 10,
        padding: int = 0,
        stride: int = 1,
        dropout: float = 0,
        n_feat: int = 500,
        init_neur: int = 100,
        n_neur: int = 10,
        n_out: int = 1,
    ) -> None:
        super(AnomNet, self).__init__()
        self.Flatten = Flatten()

        self.fc0 = nn.Linear(n_feat, n_neur)
        self.fc1 = nn.Linear(init_neur, n_neur)
        self.fc2 = nn.Linear(n_neur, n_neur)
        self.fc_out = nn.Linear(n_neur, n_out)
        self.conv1a = nn.Conv1d(
            in_channels=init_filt,
            out_channels=n_filt,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv1b = nn.Conv1d(
            in_channels=n_filt,
            out_channels=n_filt,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv1c = nn.Conv1d(
            in_channels=n_filt,
            out_channels=n_filt,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the AnomNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, init_filt, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_out).
        """
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        # x = F.relu(self.conv1c(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc_out(x))
        x = self.fc_out(x)
        return x
