"""
Preprocessing functions for Pubmed dataset
"""

import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

from utils import set_seed, load_params_from_file, get_file_number_of_lines, save_to_pickle

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def _process_pubmed_nodes_labels(file_path, pubmed_nodes):
    file_num_lines = get_file_number_of_lines(file_path)

    nodes_file = open(file_path, 'r')
    for line_idx, line in tqdm(enumerate(nodes_file), total=file_num_lines, desc="Processing labels"):
        if line_idx < 3:
            # Skip header and template line
            continue

        node_id = line.split("\t")[0].strip().rstrip()
        node_label = int(line.split("\t")[1].split("label=")[1].strip().rstrip())
        pubmed_nodes[node_id] = {"label": node_label}


def _process_pubmed_nodes_metadata(file_path, pubmed_nodes):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    for node_data in tqdm(json_data, desc="Processing metadata"):
        node_id = node_data.get("PMID")
        if node_id and node_id in pubmed_nodes:
            pubmed_nodes[node_id]["title"] = node_data["TI"]
            pubmed_nodes[node_id]["abstract"] = node_data["AB"]


def _process_pubmed_nodes_edges(file_path, pubmed_nodes):
    file_num_lines = get_file_number_of_lines(file_path)

    edges_file = open(file_path, 'r')
    for line_idx, line in tqdm(enumerate(edges_file), total=file_num_lines, desc="Processing edges"):
        if line_idx < 3:
            # Skip header and template line
            continue

        source_node = line.split("\t")[1].strip().rstrip().split("paper:")[1]
        target_node = line.split("\t")[3].strip().rstrip().split("paper:")[1]
        if source_node in pubmed_nodes and target_node in pubmed_nodes:
            # Outgoing edges
            if "edges" not in pubmed_nodes[source_node]:
                pubmed_nodes[source_node]["edges"] = [target_node]
            else:
                pubmed_nodes[source_node]["edges"].append(target_node)

            # And ingoing edges
            if "edges" not in pubmed_nodes[target_node]:
                pubmed_nodes[target_node]["edges"] = [source_node]
            else:
                pubmed_nodes[target_node]["edges"].append(source_node)


def run_preprocessing(config):
    # Format file paths
    pubmed_nodes_file_path = os.path.join(config.get('root_path'), 'pubmed', 'Pubmed-Diabetes.NODE.paper.tab')
    pubmed_edges_file_path = os.path.join(config.get('root_path'), 'pubmed', 'Pubmed-Diabetes.DIRECTED.cites.tab')
    pubmed_nodes_metadata_file_path = os.path.join(config.get('root_path'), 'pubmed', 'pubmed.json')

    pubmed_nodes = dict()

    # Process node labels
    _process_pubmed_nodes_labels(pubmed_nodes_file_path, pubmed_nodes)

    # Process node metadata (abstracts and titles)
    _process_pubmed_nodes_metadata(pubmed_nodes_metadata_file_path, pubmed_nodes)

    # Process node edge list
    _process_pubmed_nodes_edges(pubmed_edges_file_path, pubmed_nodes)

    # Save output to disk
    output_path = os.path.join(config.get('output_path'), 'pubmed_graph.pkl')
    save_to_pickle(pubmed_nodes, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/preprocessing.json",
                        required=True,
                        help='Path to the json config for preprocessing')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    preprocessing_params = load_params_from_file(args.config_path)
    logging.info(f"Parameters for preprocessing: {preprocessing_params}")

    run_preprocessing(preprocessing_params)
