"""
Module defining a trainer for HyperBERT
"""

import torch
from torch import nn

from utils import get_available_device, get_trainable_parameters, get_huggingface_tokenizer
from sklearn.metrics import accuracy_score


class FinetuningTrainer:
    def __init__(self, params, model, device=get_available_device()):
        """
        params (dict): A dict with the configuration parameters (e.g., learning rate, optimizer, etc.)
        """
        super().__init__()

        self.params = params
        self.model = model
        self.tokenizer = get_huggingface_tokenizer(params['base_model_name'])
        self.classification_loss = nn.CrossEntropyLoss()
        self.device = device

    def calculate_loss(self, y_hat, y_true):
        classification_loss = self.classification_loss(y_hat, y_true)
        return {"loss": classification_loss}

    def calculate_metrics(self, y_hat, y_true):
        y_pred = y_hat.argmax(dim=1)
        balanced_accuracy = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return {"train_balanced_accuracy": balanced_accuracy}

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

    def run_train_step(self, text, graph, y):
        """
        Returns
        - loss (torch.Tensor): The total loss, used for backpropagation
        - metrics_to_log (dict): A dictionary with the calculated metrics (detached from the computational graph)
        """
        self.model.train()

        # Tokenize node texts
        encoded_text = self.tokenizer(text,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True,
                                      max_length=512)

        # Forward pass
        hyperbert_output = self.model.encoder(**encoded_text,
                                              graph_batch=graph,
                                              return_dict=True)

        cls_embedding = hyperbert_output.pooler_output

        # Forward pass through the classifier
        y_hat = self.model.classifier(cls_embedding)

        # Calculate loss
        loss = self.calculate_loss(y_hat, y)

        # Calculate metrics
        metrics = self.calculate_metrics(y_hat, y)

        # Calculate metrics
        log_metrics = {"train_loss": loss["loss"].detach().item()}
        log_metrics.update(metrics)

        return loss["loss"], log_metrics
