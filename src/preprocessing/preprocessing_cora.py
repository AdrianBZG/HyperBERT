"""
Preprocessing functions for Cora dataset
"""

import argparse
import logging
import os
import sys
from tqdm import tqdm

from utils import set_seed, load_params_from_file, get_file_number_of_lines, save_to_pickle, hypergraph_from_graph

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def _process_cora_nodes_labels(file_path, cora_nodes):
    file_num_lines = get_file_number_of_lines(file_path)

    nodes_file = open(file_path, 'r')
    for line_idx, line in tqdm(enumerate(nodes_file), total=file_num_lines, desc="Processing labels"):
        node_id = line.split("\t")[0].strip().rstrip()
        node_label = line.split("\t")[-1].strip().rstrip()
        cora_nodes[node_id] = {"label": node_label}


def _process_cora_nodes_metadata(metadata_file_path, extractions_root_path, cora_nodes):
    file_num_lines = get_file_number_of_lines(metadata_file_path)

    metadata_file = open(metadata_file_path, 'r')
    for line_idx, line in tqdm(enumerate(metadata_file), total=file_num_lines, desc="Processing metadata"):
        node_id = line.split("\t")[0].strip().rstrip()
        if node_id not in cora_nodes:
            continue

        node_extractions_url = line.split("\t")[1].strip().rstrip()
        cora_nodes[node_id]["extractions_url"] = node_extractions_url

    for node_id in tqdm(cora_nodes.keys(), desc="Processing extractions"):
        extractions_file_path = os.path.join(extractions_root_path, f"{cora_nodes[node_id]['extractions_url']}")
        if os.path.exists(extractions_file_path):
            extractions_file = open(extractions_file_path, 'r')
            for line in enumerate(extractions_file):
                line = line[1].strip().rstrip()
                if "Title:" in line:
                    # Get the title
                    if len(line.split("Title: ")) > 1:
                        title = line.split("Title: ")[1].strip().rstrip()
                    else:
                        title = ""

                    cora_nodes[node_id]["title"] = title
                elif "Abstract:" in line:
                    # Get the abstract
                    if len(line.split("Abstract: ")) > 1:
                        abstract = line.split("Abstract: ")[1].strip().rstrip()
                    else:
                        abstract = ""

                    cora_nodes[node_id]["abstract"] = abstract

    # Clean nodes without abstracts and titles or empty ones
    nodes_to_remove = set()
    for node_id in cora_nodes.keys():
        if "abstract" not in cora_nodes[node_id] or "title" not in cora_nodes[node_id]:
            nodes_to_remove.add(node_id)
        else:
            if len(cora_nodes[node_id]["abstract"]) == 0 and len(cora_nodes[node_id]["title"]) == 0:
                nodes_to_remove.add(node_id)

    for node_id in nodes_to_remove:
        del cora_nodes[node_id]


def _process_cora_nodes_edges(file_path, cora_nodes):
    file_num_lines = get_file_number_of_lines(file_path)

    edges_file = open(file_path, 'r')
    for line_idx, line in tqdm(enumerate(edges_file), total=file_num_lines, desc="Processing edges"):
        source_node = line.split("\t")[0].strip().rstrip()
        target_node = line.split("\t")[1].strip().rstrip()
        if source_node in cora_nodes and target_node in cora_nodes:
            # Outgoing edges
            if "edges" not in cora_nodes[source_node]:
                cora_nodes[source_node]["edges"] = [target_node]
            else:
                cora_nodes[source_node]["edges"].append(target_node)

            # And ingoing edges
            if "edges" not in cora_nodes[target_node]:
                cora_nodes[target_node]["edges"] = [source_node]
            else:
                cora_nodes[target_node]["edges"].append(source_node)


def run_preprocessing(config):
    # Format file paths
    cora_nodes_file_path = os.path.join(config.get('root_path'), 'cora', 'cora.content')
    cora_edges_file_path = os.path.join(config.get('root_path'), 'cora', 'cora.cites')
    cora_nodes_metadata_file_path = os.path.join(config.get('root_path'), 'cora', 'papers')
    cora_nodes_extractions_root_path = os.path.join(config.get('root_path'), 'cora', 'extractions')

    cora_nodes = dict()

    # Process node labels
    _process_cora_nodes_labels(cora_nodes_file_path, cora_nodes)

    # Process node metadata (abstracts and titles)
    _process_cora_nodes_metadata(cora_nodes_metadata_file_path, cora_nodes_extractions_root_path, cora_nodes)

    # Process node edge list
    _process_cora_nodes_edges(cora_edges_file_path, cora_nodes)

    # Save output to disk
    output_path = os.path.join(config.get('output_path'), 'cora_graph.pkl')
    save_to_pickle(cora_nodes, output_path)


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
