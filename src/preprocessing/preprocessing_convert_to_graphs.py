"""
Preprocessing functions for Pubmed dataset
"""

import logging
import os
import pickle

import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from utils import set_seed, load_params_from_file, get_file_number_of_lines, save_to_pickle, hypergraph_from_graph, dense_to_sparse_incidence

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')

FILE_PATHS = {"cora": "data/preprocessed/cora_graph.pkl",
              "pubmed": "data/preprocessed/pubmed_graph.pkl"}


def transform_graphs_to_networkx():
    networkx_graphs = dict()

    for dataset, file_path in tqdm(FILE_PATHS.items(), desc="Transforming graphs to Networkx"):
        with open(file_path, 'rb') as file:
            graph = pickle.load(file)

        # Transform graph to networkx
        networkx_graph = nx.DiGraph()
        node_embeddings = {}
        node_labels = {}
        for source_node, node_data in tqdm(graph.items(), desc=f"Transforming {dataset} graph"):
            if not all(k in node_data for k in ("edges", "label", "title", "abstract")):
                continue

            if node_data["label"] not in node_labels:
                node_labels[node_data["label"]] = len(node_labels)

            if source_node not in networkx_graph:
                if source_node not in node_embeddings:
                    node_embeddings[source_node] = torch.rand(768)

                networkx_graph.add_node(source_node,
                                        x=node_embeddings[source_node],
                                        y=node_labels[node_data["label"]],
                                        title=node_data["title"],
                                        abstract=node_data["abstract"])

            node_edges = node_data["edges"]
            for target_node in node_edges:
                if target_node not in networkx_graph:
                    if target_node not in node_embeddings:
                        node_embeddings[target_node] = torch.rand(768)

                    networkx_graph.add_node(target_node,
                                            x=node_embeddings[target_node],
                                            y=node_labels[node_data["label"]],
                                            title=node_data["title"],
                                            abstract=node_data["abstract"])

                networkx_graph.add_edge(source_node, target_node)

        networkx_graphs[dataset] = networkx_graph

    return networkx_graphs


def get_incidency_matrices(networkx_graphs):
    incidency_matrices = dict()

    for dataset, graph in tqdm(networkx_graphs.items(), desc="Obtaining hypergraph incidency matrices"):
        incidency_matrix = hypergraph_from_graph(graph, k=1)
        sparse_indicency_matrix = dense_to_sparse_incidence(incidency_matrix)
        incidency_matrices[dataset] = sparse_indicency_matrix

    return incidency_matrices


def transform_networkx_to_pyg(networkx_graphs, incidency_matrices):
    pyg_graphs = dict()

    for dataset, graph in tqdm(networkx_graphs.items(), desc="Transforming graphs to PyG format"):
        pyg_graph = from_networkx(graph)
        pyg_graph.hyperedge_index = incidency_matrices[dataset]
        pyg_graphs[dataset] = pyg_graph

    return pyg_graphs


if __name__ == '__main__':
    set_seed()

    # Check file paths
    for file_path in FILE_PATHS.values():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

    # Obtain Networkx graphs for each dataset
    networkx_graphs = transform_graphs_to_networkx()

    # Obtain incidency matrix for the hypergraphs
    incidency_matrices = get_incidency_matrices(networkx_graphs)

    # Save incidency matrices to disk
    output_path = os.path.join('data', 'preprocessed', 'cora_incidency_matrix.pkl')
    save_to_pickle(incidency_matrices["cora"], output_path)
    output_path = os.path.join('data', 'preprocessed', 'pubmed_incidency_matrix.pkl')
    save_to_pickle(incidency_matrices["pubmed"], output_path)

    # Obtain PyG graphs for each dataset
    pyg_graphs = transform_networkx_to_pyg(networkx_graphs, incidency_matrices)

    # Save PyG graphs to disk
    output_path = os.path.join('data', 'preprocessed', 'cora_pyg.pkl')
    save_to_pickle(pyg_graphs["cora"], output_path)
    output_path = os.path.join('data', 'preprocessed', 'pubmed_pyg.pkl')
    save_to_pickle(pyg_graphs["pubmed"], output_path)
