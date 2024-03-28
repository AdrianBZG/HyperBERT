"""
Defines the GNN layer for HyperBert
"""

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import HypergraphConv

from transformers.utils import logging

from utils import get_available_device

logger = logging.get_logger(__name__)


class HyperBertGNNLayer(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, config.hidden_size)
        self.conv2 = GCNConv(config.hidden_size, config.hidden_size)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x


class HyperBertHGNNLayer(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.conv1 = HypergraphConv(in_features, config.hidden_size)
        self.conv2 = HypergraphConv(config.hidden_size, config.hidden_size)

    def forward(self, graph):
        x, hyperedge_index = graph["x"], graph["hyperedge_index"]

        x = self.conv1(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, hyperedge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x
