#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import sys

import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from utils import set_seed, save_to_pickle, hypergraph_from_graph, dense_to_sparse_incidence

logging.basicConfig(level=logging.INFO,
                    format='[%(name)s:%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Define file paths for available datasets.
FILE_PATHS = {
    "cora_co": "data/hyperbert/cora_co/cora_co_graph.pkl",
    "pubmed": "data/hyperbert/pubmed/pubmed_graph.pkl",
    "dblp_a": "data/hyperbert/dblp_a/dblp_graph.pkl",
    "imdb": "data/hyperbert/imdb/imdb_graph.pkl"
}

def transform_graphs_to_networkx(file_paths):
    """
    Transform the graphs stored in the provided file_paths dictionary into Networkx graphs.
    
    For dblp_a: expects keys "edges", "labels", "title", "abstract".
    For cora_co and pubmed: expects keys "edges", "label", "title", "abstract".
    For imdb: expects keys "edges", "label", "title", "actors" (optionally "year" and "runtime").
    """
    networkx_graphs = dict()
    for dataset, file_path in tqdm(file_paths.items(), desc="Transforming graphs to Networkx"):
        with open(file_path, 'rb') as file:
            graph = pickle.load(file)

        # Create a directed Networkx graph.
        networkx_graph = nx.DiGraph()
        node_embeddings = {}
        node_labels = {}
        for source_node, node_data in tqdm(graph.items(), desc=f"Processing nodes for {dataset}", leave=False):
            # Branch based on dataset.
            if dataset == "dblp_a":
                if not all(k in node_data for k in ("edges", "labels", "title", "abstract")):
                    continue
                label_val = node_data["labels"][0] if node_data["labels"] else "unknown"
            elif dataset == "imdb":
                if not all(k in node_data for k in ("edges", "label", "title", "actors")):
                    continue
                label_val = node_data["label"]
            else:
                if not all(k in node_data for k in ("edges", "label", "title", "abstract")):
                    continue
                label_val = node_data["label"]

            # Map label to an integer.
            if label_val not in node_labels:
                node_labels[label_val] = len(node_labels)
            int_label = node_labels[label_val]

            # Add the source node if not already present.
            if source_node not in networkx_graph:
                if source_node not in node_embeddings:
                    node_embeddings[source_node] = torch.rand(768)
                # Build a node attribute dictionary based on dataset.
                if dataset == "imdb":
                    networkx_graph.add_node(source_node,
                                             x=node_embeddings[source_node],
                                             y=int_label,
                                             title=node_data["title"],
                                             actors=node_data.get("actors", []),
                                             year=node_data.get("year", None),
                                             runtime=node_data.get("runtime", None))
                else:
                    networkx_graph.add_node(source_node,
                                             x=node_embeddings[source_node],
                                             y=int_label,
                                             title=node_data["title"],
                                             abstract=node_data["abstract"])
            # Process outgoing edges.
            node_edges = node_data["edges"]
            for target_node in node_edges:
                if target_node not in networkx_graph:
                    # Try to fetch metadata from the original graph.
                    if target_node in graph:
                        target_data = graph[target_node]
                        if dataset == "dblp_a":
                            target_label_val = target_data["labels"][0] if target_data.get("labels") else "unknown"
                        elif dataset == "imdb":
                            if not all(k in target_data for k in ("label", "title", "actors")):
                                continue
                            target_label_val = target_data.get("label", "unknown")
                        else:
                            target_label_val = target_data.get("label", "unknown")
                        if target_label_val not in node_labels:
                            node_labels[target_label_val] = len(node_labels)
                        target_int_label = node_labels[target_label_val]
                        if dataset == "imdb":
                            node_title = target_data.get("title", "")
                            node_actors = target_data.get("actors", [])
                            node_year = target_data.get("year", None)
                            node_runtime = target_data.get("runtime", None)
                        else:
                            node_title = target_data.get("title", "")
                            node_abstract = target_data.get("abstract", "")
                    else:
                        target_int_label = int_label
                        if dataset == "imdb":
                            node_title = ""
                            node_actors = []
                            node_year = None
                            node_runtime = None
                        else:
                            node_title = ""
                            node_abstract = ""
                    if target_node not in node_embeddings:
                        node_embeddings[target_node] = torch.rand(768)
                    if dataset == "imdb":
                        networkx_graph.add_node(target_node,
                                                 x=node_embeddings[target_node],
                                                 y=target_int_label,
                                                 title=node_title,
                                                 actors=node_actors,
                                                 year=node_year,
                                                 runtime=node_runtime)
                    else:
                        networkx_graph.add_node(target_node,
                                                 x=node_embeddings[target_node],
                                                 y=target_int_label,
                                                 title=node_title,
                                                 abstract=node_abstract)
                networkx_graph.add_edge(source_node, target_node)
        networkx_graphs[dataset] = networkx_graph
    return networkx_graphs

def get_incidency_matrices(networkx_graphs):
    incidency_matrices = dict()
    for dataset, graph in tqdm(networkx_graphs.items(), desc="Obtaining hypergraph incidency matrices"):
        incidency_matrix = hypergraph_from_graph(graph, k=1)
        sparse_incidence_matrix = dense_to_sparse_incidence(incidency_matrix)
        incidency_matrices[dataset] = sparse_incidence_matrix
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
    parser = argparse.ArgumentParser(description="Transform graphs to PyG format.")
    parser.add_argument('--datasets', type=str, default=",".join(FILE_PATHS.keys()),
                        help="Comma-separated list of dataset names to process (keys in FILE_PATHS)")
    args = parser.parse_args(sys.argv[1:])

    # Filter FILE_PATHS to only include the selected datasets.
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip() in FILE_PATHS]
    if not selected_datasets:
        raise ValueError("No valid dataset names provided. Valid options: " + ", ".join(FILE_PATHS.keys()))
    selected_file_paths = {d: FILE_PATHS[d] for d in selected_datasets}

    # Check that each file exists.
    for dataset, file_path in selected_file_paths.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File for dataset {dataset} not found: {file_path}")

    # Transform the selected graphs into Networkx format.
    networkx_graphs = transform_graphs_to_networkx(selected_file_paths)

    # Obtain hypergraph incidence matrices.
    incidency_matrices = get_incidency_matrices(networkx_graphs)

    # For each dataset, display the number of nodes, hyperedges, and classes.
    for ds in selected_datasets:
        nx_graph = networkx_graphs[ds]
        num_nodes = nx_graph.number_of_nodes()
        # Compute unique classes from the 'y' attribute.
        classes = set()
        for _, data in nx_graph.nodes(data=True):
            classes.add(data["y"])
        num_classes = len(classes)
        # For sparse incidence matrices, assume incidence is a tuple (node_indices, hyperedge_indices).
        incidence = incidency_matrices[ds]
        node_indices, hyperedge_indices = incidence
        if hyperedge_indices.numel() == 0:
            num_hyperedges = 0
        else:
            num_hyperedges = int(hyperedge_indices.max().item()) + 1
        logger.info(f"Dataset: {ds} - Nodes: {num_nodes}, Hyperedges: {num_hyperedges}, Classes: {num_classes}")

    # Optionally, save incidence matrices for the selected datasets.
    for ds in selected_datasets:
        out_path = os.path.join('data', 'hyperbert', ds, f'{ds}_incidency_matrix.pkl')
        save_to_pickle(incidency_matrices[ds], out_path)
        logger.info(f"Saved {ds} incidence matrix to {out_path}")

    # Transform the Networkx graphs to PyG graphs.
    pyg_graphs = transform_networkx_to_pyg(networkx_graphs, incidency_matrices)

    # Save the PyG graphs to disk.
    for ds in selected_datasets:
        out_path = os.path.join('data', 'hyperbert', ds, f'{ds}_pyg.pkl')
        save_to_pickle(pyg_graphs[ds], out_path)
        logger.info(f"Saved {ds} PyG graph to {out_path}")