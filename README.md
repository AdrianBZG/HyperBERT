# HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs
### [Paper](https://aclanthology.org/2024.findings-emnlp.537/)

This is the official implementation and dataset of the paper "HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs", published at EMNLP 2024.

<img src="https://github.com/AdrianBZG/HyperBERT/blob/add_data/figures/hyperbert.png" width="750" alt="HyperBERT Image">

## Data

All preprocessed data is located in the [data/hyperbert directory](data/hyperbert directory). This folder contains preprocessed PyTorch Geometric (PyG) graph files (stored as pickle files) for the benchmark datasets used in HyperBERT. The available datasets include:
- [cora_co](data/cora_co): The Cora co-citation hypergraph.
- [dblp_a](data/dblp_a): The DBLP academic publications hypergraph.
- [imdb](data/imdb): The IMDB movie hypergraph.
- [pubmed](data/pubmed): The PubMed citation hypergraph.

Each dataset is stored in a file named <dataset_name>_pyg.pkl (for example, imdb_pyg.pkl). These PyG graphs include the following key components:
- Node attributes:
  - title and abstract for datasets like cora_co, dblp_a, and pubmed.
  - For the imdb dataset, nodes include the title (movie title) along with additional attributes such as actors, year, and runtime.
- hyperedge_index: A sparse representation of the hypergraph incidence matrix.
- y: A tensor of node labels.

For a hands-on example of how to load these datasets for training or inspection, see the [demo_load_hypergraph_datasets.py](src/demo_load_hypergraph_datasets.py) script. The script demonstrates how to use our get_data_loader function to create a PyTorch DataLoader from a preprocessed dataset which can be then used for training models.

## Citation

```
@inproceedings{bazaga-etal-2024-hyperbert,
    title = "{H}yper{BERT}: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs",
    author = "Bazaga, Adri{\'a}n  and
      Lio, Pietro  and
      Micklem, Gos",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.537/",
    doi = "10.18653/v1/2024.findings-emnlp.537"
}
```

## Contact

For feedback, questions, or press inquiries please contact [Adrián Bazaga](mailto:ar989@cam.ac.uk)
