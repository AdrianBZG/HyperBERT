"""
Module defining the losses used for training.
"""
import torch
import torch.nn as nn
from pytorch_metric_learning import losses


def adjust_labels(labels, previous_max_label):
    labels += previous_max_label + 1
    enqueue_mask = torch.zeros(len(labels)).bool()
    enqueue_mask[labels.mask()] = True
    return labels, enqueue_mask


class StructuralLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(StructuralLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = losses.SupConLoss(temperature=self.temperature)

    def forward(self, graph_states) -> torch.float:
        t_loss = 0.0
        batch_size = len(graph_states)
        t_number_node_states = sum(graph_node_states.shape[0] for graph_node_states in graph_states)

        for b_id in range(0, batch_size):
            num_positives = graph_states[b_id].size(0)
            num_negatives = t_number_node_states - num_positives
            labels = torch.arange(0, num_positives + num_negatives)
            labels[:num_positives] = 0

            positive_embeddings = graph_states[b_id]
            negative_embeddings = [graph_states[i] for i in range(len(graph_states)) if i != b_id]
            negative_embeddings = torch.cat(negative_embeddings, dim=0)
            all_embeddings_ordered = torch.cat([positive_embeddings,
                                                negative_embeddings], dim=0)

            loss = self.loss_func(all_embeddings_ordered, labels)

            t_loss += loss

        t_loss /= batch_size
        return t_loss


class SemanticLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SemanticLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = losses.SupConLoss(temperature=self.temperature)

    def forward(self, semantic_states) -> torch.float:
        batch_size = semantic_states.shape[0]
        labels = torch.arange(0, batch_size)
        loss = self.loss_func(semantic_states, labels)
        return loss


class AlignmentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(AlignmentLoss, self).__init__()
        self.temperature = temperature
        self.loss_func = losses.SupConLoss(temperature=self.temperature)

    def forward(self, graph_states, semantic_states) -> torch.float:
        pooled_graph_states = torch.stack([graph_state.mean(0) for graph_state in graph_states], dim=0)
        all_states = torch.cat([pooled_graph_states, semantic_states], dim=0)
        labels = torch.cat((torch.arange(0, pooled_graph_states.shape[0]),
                            torch.arange(0, semantic_states.shape[0])), dim=0)
        loss = self.loss_func(all_states, labels)
        return loss
