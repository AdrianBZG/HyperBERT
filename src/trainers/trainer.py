"""
Module defining a trainer for HyperBERT
"""
import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from utils import get_available_device, get_trainable_parameters, get_huggingface_tokenizer
from losses import SemanticLoss, StructuralLoss, AlignmentLoss
from models.hyperbert_graph import HyperBertGraphModelOutput
from dataset.hyperbert_dataset import HyperBertDataset


def batch_collate_function(batch):
    text = [item['text'] for item in batch]
    y = torch.tensor([item['y'] for item in batch], dtype=torch.long, device=get_available_device())
    node_idx = [item['node_idx'] for item in batch]
    graph = [Data(x=item['graph_x'],
                  hyperedge_index=item['graph_hyperedge_index'],
                  node_idx=item['node_idx'])
             for item in batch]

    return {"text": text,
            "y": y,
            "graph": graph,
            "node_idx": node_idx}


def get_data_loader(config):
    pickle_path = os.path.join(config['data_base_path'],
                               f'{config["dataset_name"]}_pyg.pkl')

    with open(pickle_path, 'rb') as file:
        pyg_graph = pickle.load(file)
        del pyg_graph.edge_index

    dataset = HyperBertDataset(pyg_graph)
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            collate_fn=lambda batch: batch_collate_function(batch))

    return dataloader


class Trainer:
    def __init__(self, params, model, device=get_available_device()):
        """
        params (dict): A dict with the configuration parameters (e.g., learning rate, optimizer, etc.)
        """
        super().__init__()

        self.params = params
        self.model = model
        self.tokenizer = get_huggingface_tokenizer(params['base_model_name'], device=device)
        self.semantic_loss = SemanticLoss(temperature=params['loss_temperature'])
        self.structural_loss = StructuralLoss(temperature=params['loss_temperature'])
        self.alignment_loss = AlignmentLoss(temperature=params['loss_temperature'])
        self.device = device

        # Move to device
        self.model.to(device)

    def calculate_loss(self, hyperbert_output: HyperBertGraphModelOutput):
        structural_loss = self.structural_loss(hyperbert_output.graph_hidden_states)
        semantic_loss = self.semantic_loss(hyperbert_output.semantic_hidden_states)
        alignment_loss = self.alignment_loss(hyperbert_output.graph_hidden_states,
                                             hyperbert_output.semantic_hidden_states)

        total_loss = structural_loss + semantic_loss + alignment_loss

        return {"structural_loss": structural_loss,
                "semantic_loss": semantic_loss,
                "alignment_loss": alignment_loss,
                "loss": total_loss}

    def prepare_optimizers(self):
        params = self.model.parameters()
        print(f"The model has {get_trainable_parameters(self.model)} trainable parameters")

        optimizer_type = self.params['optimizer']
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(list(params),
                                          lr=self.params['learning_rate'])
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(list(params),
                                         lr=self.params['learning_rate'], eps=1e-7)
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(list(params),
                                        lr=self.params['learning_rate'])
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer

    def run_train_step(self, text, graph):
        """
        Returns
        - loss (torch.Tensor): The total loss, used for backpropagation
        - metrics_to_log (dict): A dictionary with the calculated metrics (detached from the computational graph)
        """
        self.model.train()

        # Place graphs on model device
        graph = [grp.to(self.model.device) for grp in graph]

        # Tokenize node texts
        encoded_text = self.tokenizer(text,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True,
                                      max_length=self.params['max_seq_length']).to(self.model.device)

        # Forward pass
        hyperbert_output = self.model(**encoded_text,
                                      graph_batch=graph,
                                      return_dict=True)

        # Calculate loss
        loss = self.calculate_loss(hyperbert_output)

        # Calculate metrics
        log_metrics = {"train_loss": loss["loss"].detach().item()}
        return loss["loss"], log_metrics
