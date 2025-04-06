"""
Module to wrap a HyperBertDataset
"""

import logging

import torch
from torch.utils.data import Dataset
from utils import get_sub_hypergraph

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(f"{__name__}")


class HyperBertDataset(Dataset):
    def __init__(self, pyg_graph, ds_name):
        self.ds_name = ds_name
        assert self.ds_name in ["cora_co", "dblp_a", "imdb", "pubmed"]
        self.graph = pyg_graph
        
        if self.ds_name == "imdb":
            # For IMDB, we only have 'title' (no abstract)
            self.text = list(pyg_graph.title)
        else:
            # For other datasets, concatenate abstract and title.
            self.text = [abstract + " " + title for abstract, title in zip(pyg_graph.abstract, pyg_graph.title)]

        self.y = [y.item() for y in pyg_graph.y]

        # Safety checks
        assert len(self.text) == len(self.y)

        print(f"Loaded HyperBertDataset with {len(self.text)} samples")

    def __getitem__(self, idx):
        sub_hypergraph = get_sub_hypergraph(hypergraph=self.graph,
                                            center_node_idx=idx)
        text = self.text[idx]
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return {"graph_x": sub_hypergraph.x,
                "graph_hyperedge_index": sub_hypergraph.hyperedge_index,
                "text": text,
                "y": y,
                "node_idx": sub_hypergraph.node_idx}

    def __len__(self):
        return len(self.text)
