import json
import os
import pickle
import torch
import random
import logging
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

GLOBAL_SEED = 498237453
logging.basicConfig(level=logging.INFO,
                    format='[utils:%(levelname)s] %(message)s')


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dummy_graph_dataset():
    dataset = Planetoid(root='./data/dummy', name='Cora')
    return dataset


def freeze_model(model_to_freeze):
    for param in model_to_freeze.parameters():
        param.requires_grad = False


def get_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


def get_available_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"  # Many issues with MPS so will force CPU here for now
    else:
        device = "cpu"

    return device


def get_huggingface_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.is_fast:
        raise ValueError('Only fast tokenizers are supported.')

    model = AutoModel.from_pretrained(model_name).to(get_available_device())

    return {"tokenizer": tokenizer,
            "model": model}


def get_huggingface_tokenizer(model_name, device=get_available_device()):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device=device)

    if not tokenizer.is_fast:
        raise ValueError('Only fast tokenizers are supported.')

    return tokenizer


def get_file_number_of_lines(file_path):
    with open(file_path) as fp:
        num_lines = sum(1 for _ in fp)

    return num_lines


def save_to_pickle(data, save_path):
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_from_checkpoint(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f'Model checkpoint does not exist at {model_path}')

    model = torch.load(model_path)
    return model


def load_params_from_file(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f'Path to config {file_path} does not exist.')

    logging.info(f"Loading parameters from: {file_path}")
    with open(file_path) as file:
        try:
            params = json.load(file)
        except Exception as e:
            logging.error(e)

    return params


def find_k_hop_neighbors(graph: nx.DiGraph, k=1):
    """
    For each node in the graph, return a dictionary mapping the node
    to the set of nodes that are reachable in exactly k hops.

    Parameters:
        graph (nx.DiGraph): The input directed graph.
        k (int): The number of hops to consider.

    Returns:
        dict: A dictionary where keys are nodes and values are sets of nodes
              that are reachable from the key node in exactly k hops.
    """
    k_hop_neighbors = {}
    
    # Iterate over all nodes in the graph.
    for node in graph.nodes():
        current_hop = {node}  # Start with the node itself.
        # Iteratively compute neighbors for each hop.
        for _ in range(k):
            next_hop = set()
            for n in current_hop:
                next_hop.update(graph.successors(n))
            current_hop = next_hop
        # Optionally, remove the original node if present.
        current_hop.discard(node)
        k_hop_neighbors[node] = current_hop

    return k_hop_neighbors


def dense_to_sparse_incidence(incidence_matrix):
    """
    Transforms a dense hypergraph incidence matrix to a sparse representation.

    Parameters:
    - incidence_matrix: 2D numpy array representing the dense incidence matrix of a hypergraph.

    Returns:
    A tuple of two lists, where the second list contains the hyperedge indices and the first list contains
    the node indices.
    """
    # Initialize the sparse matrix components
    hyperedge_indices = []

    # Iterate through the matrix to find non-zero elements
    for node_id, node_hyperedge_membership in enumerate(incidence_matrix):
        for hyperedge_id, hyperedge_flag in enumerate(node_hyperedge_membership):
            if hyperedge_flag == 1:
                node_hyperedge_pair = (node_id, hyperedge_id)
                if node_hyperedge_pair not in hyperedge_indices:
                    hyperedge_indices.append(node_hyperedge_pair)

    hyperedge_indices = torch.tensor([
        [pair[0] for pair in hyperedge_indices],
        [pair[1] for pair in hyperedge_indices]
    ], dtype=torch.long)

    return hyperedge_indices


def hypergraph_from_graph(graph: nx.DiGraph, k=1):
    # Step 1: Identify k-hop neighbors for each node
    k_hop_neighbors = find_k_hop_neighbors(graph, k)

    # Step 2: Create unique hyperedges using frozensets for deduplication
    unique_hyperedges = set()
    for node, neighbors in tqdm(k_hop_neighbors.items(), desc="Creating unique hyperedges"):
        if neighbors:  # Only consider non-empty neighbor sets
            unique_hyperedges.add(frozenset(neighbors))
    hyperedges = list(unique_hyperedges)

    # Step 3: Build the hypergraph incidence matrix (nodes x hyperedges)
    num_nodes = len(graph.nodes())
    num_hyperedges = len(hyperedges)
    H = np.zeros((num_nodes, num_hyperedges))

    node_list = list(graph.nodes())  # Create a consistent ordering of nodes
    for j, hyperedge in tqdm(enumerate(hyperedges), desc="Building hypergraph incidence matrix"):
        for node in hyperedge:
            if node in node_list:  # Sanity check (should be always true)
                i = node_list.index(node)
                H[i, j] = 1  # Set membership to 1

    return H


@torch.no_grad()
def get_sub_hypergraph(hypergraph, center_node_idx):
    """
    Gets the subhypergraph of a given node by index.
    """
    # Extract node and hyperedge indices
    hyperedge_index = hypergraph.hyperedge_index
    node_indices = hyperedge_index[0, :]
    hyperedge_indices = hyperedge_index[1, :]

    # Find hyperedges that the specified node belongs to
    relevant_hyperedges = hyperedge_indices[torch.where(node_indices == center_node_idx)]

    # Filter for the nodes that are in the relevant hyperedges
    mask = torch.isin(hyperedge_indices, relevant_hyperedges)
    filtered_nodes = node_indices[mask]

    unique_vals, filtered_nodes = torch.unique(filtered_nodes, return_inverse=True)
    new_center_node_idx = filtered_nodes[torch.where(unique_vals == center_node_idx)]
    filtered_node_attributes = hypergraph.x[unique_vals]
    filtered_hyperedges = hyperedge_indices[mask]

    # Combine them back into a sparse edge list
    sub_hyperedge_index = torch.vstack((filtered_nodes, filtered_hyperedges))

    # Create the subhypergraph object
    subhypergraph = Data(hyperedge_index=sub_hyperedge_index,
                         x=filtered_node_attributes,
                         node_idx=new_center_node_idx)

    return subhypergraph


def get_graph_input_dim_for_dataset(dataset_name):
    if dataset_name == 'cora':
        return 768
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')


def get_num_classes_for_dataset(dataset_name):
    if dataset_name == 'cora':
        return 7
    elif dataset_name == 'pubmed':
        return 3
    elif dataset_name == 'dblp-a':
        return 6
    elif dataset_name == 'cora-ca':
        return 7
    elif dataset_name == 'imdb':
        return 3
    else:
        raise ValueError(f'Unknown dataset name: {dataset_name}')


def get_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
