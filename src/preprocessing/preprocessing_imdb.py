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

def load_processed_movies(file_path):
    """
    Load nodes from a processed JSONL file.
    Each line is a JSON object representing a movie with fields:
      - id, title, actors, year, runtime, label, and references.
    The function builds a dictionary mapping movie IDs to metadata.
    """
    movies_graph = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading nodes"):
            try:
                movie = json.loads(line)
            except Exception as e:
                logger.warning(f"Skipping line due to error: {e}")
                continue
            movie_id = movie.get("id")
            if not movie_id:
                continue
            movies_graph[movie_id] = {
                "title": movie.get("title", ""),
                "actors": movie.get("actors", []),
                "year": movie.get("year", None),
                "runtime": movie.get("runtime", None),
                "label": movie.get("label", ""),
                "edges": []  # Will be filled later.
            }
            # Temporarily store references to process edges.
            movies_graph[movie_id]["_references"] = movie.get("references", [])
    return movies_graph

def process_edges(movies_graph):
    """
    Process edges for the movies graph based on the _references field.
    For every movie, if a reference exists as a node in the graph,
    add an edge from the source to the target and vice versa.
    """
    for movie_id, data in tqdm(movies_graph.items(), desc="Processing edges"):
        for ref in data.get("_references", []):
            if ref in movies_graph:
                # Add an outgoing edge from movie_id to the referenced movie.
                if ref not in data["edges"]:
                    data["edges"].append(ref)
                # Add an incoming edge to the referenced movie.
                if movie_id not in movies_graph[ref]["edges"]:
                    movies_graph[ref]["edges"].append(movie_id)
        # Remove the temporary _references key.
        data.pop("_references", None)

def save_graph(movies_graph, output_path):
    """
    Save the movies graph to disk as a pickle file.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(movies_graph, f)
    logger.info(f"Graph saved to {output_path}")

def run_preprocessing(input_file, output_file):
    logger.info("Loading processed movies...")
    movies_graph = load_processed_movies(input_file)
    logger.info(f"Loaded {len(movies_graph)} movies.")

    logger.info("Processing edges...")
    process_edges(movies_graph)

    save_graph(movies_graph, output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Build a graph from imdb.jsonl"
    )
    parser.add_argument('--input_file', required=True,
                        help='Path to the imdb.jsonl file')
    parser.add_argument('--output_file', required=True,
                        help='Path to save the graph pickle file')
    args = parser.parse_args(sys.argv[1:])

    run_preprocessing(args.input_file, args.output_file)

if __name__ == '__main__':
    main()