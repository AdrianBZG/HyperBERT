#!/usr/bin/env python3
import os
import pickle
import logging
from trainers.trainer import get_data_loader  # Assumes this function is defined as provided

logging.basicConfig(level=logging.INFO,
                    format='[%(name)s:%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Base path where your preprocessed PyG pickle files are stored.
    base_path = os.path.join("data", "hyperbert")
    # List of dataset names to process.
    dataset_names = ["cora_co", "dblp_a", "imdb", "pubmed"]

    for ds in dataset_names:
        logger.info(f"Loading dataset: {ds}")
        config = {
            "data_base_path": base_path,
            "dataset_name": ds,
            "batch_size": 1
        }
        
        # Load the underlying PyG graph to compute statistics.
        # Adjust the pickle path if your files are directly under base_path.
        pickle_path = os.path.join(config["data_base_path"], config["dataset_name"], f'{config["dataset_name"]}_pyg.pkl')
        try:
            with open(pickle_path, 'rb') as file:
                pyg_graph = pickle.load(file)
        except Exception as e:
            logger.error(f"Error loading pickle for {ds}: {e}")
            continue

        # Compute statistics.
        num_nodes = pyg_graph.x.shape[0] if hasattr(pyg_graph, "x") else 0
        num_hyperedges = pyg_graph.hyperedge_index.shape[1] if hasattr(pyg_graph, "hyperedge_index") else 0
        if hasattr(pyg_graph, "y"):
            num_labels = len(set(int(y.item()) for y in pyg_graph.y))
        else:
            num_labels = 0
        logger.info(f"Dataset: {ds} - Nodes: {num_nodes}, Hyperedges: {num_hyperedges}, Labels: {num_labels}")

        # Load the dataloader using your get_data_loader function.
        try:
            dataloader = get_data_loader(config)
        except Exception as e:
            logger.error(f"Error loading dataloader for {ds}: {e}")
            continue

        # Show one dataset entry.
        logger.info(f"Dataset {ds}: printing one sample entry:")
        entry = next(iter(dataloader))
        print("Node index:", entry["node_idx"])
        print("Text:", entry["text"])
        print("Label:", entry["y"])

if __name__ == "__main__":
    main()