"""
Module defining a simple classifier
"""

import torch
import torch.nn as nn
from utils import get_available_device


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=1, device=get_available_device()):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, out_features=self.num_classes)
        )

        self.init_params()

        self.device = device
        self.to(self.device)

    def forward(self, x, as_probabilities=False):
        """
        Returns logits if as_probabilities is False, otherwise returns probabilities.
        That's because it's more numerically stable to compute the softmax in the loss function (it uses the log-sum-exp trick)
        """
        x = x.to(self.device)
        x = self.net(x)

        if as_probabilities:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)

        return x

    def classify(self, x, as_probabilities=False):
        return self.forward(x, as_probabilities=as_probabilities)

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(self.net[0].weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
