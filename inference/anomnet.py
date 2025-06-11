"""This is a pared-down version of the AnomNet model from the original code,
where only the 1D convolutional layers and fully connected layers are retained.
https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_NN_utils.py
"""

from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AnomNet(nn.Module):
    def __init__(
        self,
        init_filt=5,
        n_filt=5,
        kernel_size=10,
        padding=0,
        stride=1,
        dropout=0,
        n_feat=500,
        init_neur=100,
        n_neur=10,
        n_out=1,
    ):
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

    def forward(self, x):
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
