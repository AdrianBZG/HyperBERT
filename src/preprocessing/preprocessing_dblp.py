#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(name)s:%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def load_processed_papers(file_path):
    """
    Load nodes from a processed JSONL file.
    Each line is a JSON object representing a paper with fields:
    id, title, authors, venue, year, abstract, labels, and references.
    The function builds a dictionary mapping paper IDs to metadata.
    """
    papers_graph = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading nodes"):
            try:
                paper = json.loads(line)
            except Exception as e:
                logger.warning(f"Skipping line due to error: {e}")
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue
            papers_graph[paper_id] = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", []),
                "venue": paper.get("venue", ""),
                "year": paper.get("year", ""),
                "labels": paper.get("labels", []),
                "edges": []  # will be filled later
            }
            # Temporarily store references to process edges.
            papers_graph[paper_id]["_references"] = paper.get("references", [])
    return papers_graph

def process_edges(papers_graph):
    """
    Process edges for the papers graph based on the _references field.
    For every paper, if a reference exists as a node in the graph,
    add an edge from the source to the target and vice versa.
    """
    for paper_id, data in tqdm(papers_graph.items(), desc="Processing edges"):
        for ref in data.get("_references", []):
            if ref in papers_graph:
                # Add an outgoing edge from paper_id to the referenced paper.
                if ref not in data["edges"]:
                    data["edges"].append(ref)
                # Add an incoming edge to the referenced paper.
                if paper_id not in papers_graph[ref]["edges"]:
                    papers_graph[ref]["edges"].append(paper_id)
        # Remove the temporary _references key.
        data.pop("_references", None)

def save_graph(papers_graph, output_path):
    """
    Save the papers graph to disk as a pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(papers_graph, f)
    logger.info(f"Graph saved to {output_path}")

def run_preprocessing(input_file, output_file):
    # Load the nodes from dblp.jsonl.
    logger.info("Loading processed papers...")
    papers_graph = load_processed_papers(input_file)
    logger.info(f"Loaded {len(papers_graph)} papers.")

    # Process edges (i.e. references).
    logger.info("Processing edges...")
    process_edges(papers_graph)

    # Save the resulting graph.
    save_graph(papers_graph, output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Build a graph from dblp.jsonl")
    parser.add_argument('--input_file',
                        required=True,
                        help='Path to the dblp.jsonl file')
    parser.add_argument('--output_file',
                        required=True,
                        help='Path to save the graph pickle file')
    args = parser.parse_args(sys.argv[1:])

    run_preprocessing(args.input_file, args.output_file)

if __name__ == '__main__':
    main()