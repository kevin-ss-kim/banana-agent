import torch
from torch import nn

class QNetwork(nn.Module):
    # QNetwork written in Pytorch

    def __init__(self, state_size, action_size, seed):
        # Initialize model
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)