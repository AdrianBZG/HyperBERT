"""
Main training entrypoint
"""
import argparse
import os
import logging
import sys

import torch
import wandb
from tqdm import tqdm, trange

from trainers.trainer import Trainer, get_data_loader
from utils import set_seed, get_graph_input_dim_for_dataset, load_params_from_file, get_available_device
from models.hyperbert_hypergraph import HyperBertHypergraphModel


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def run_train_epoch(dataloader, trainer: Trainer, optimizer):
    trainer.model.train()

    metrics_per_epoch = []
    dataloader_iterator = tqdm(dataloader, desc="Iterating batches", leave=False)
    for batch_idx, batch in enumerate(dataloader_iterator):
        text, graph = batch['text'], batch['graph']

        loss, log_metrics_dict = trainer.run_train_step(text, graph)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
        optimizer.step()

        dataloader_iterator.set_description(f"Iterating batches (Batch Idx: {batch_idx+1} | Loss: {log_metrics_dict['train_loss']:.5g})")
        dataloader_iterator.refresh()

        # Keep track of metrics
        metrics_per_epoch.append(log_metrics_dict)

    # === Aggregate metrics across iterations in the epoch ===
    metrics_names = metrics_per_epoch[0].keys()
    metrics_agg = {f"train/{metric_name}": sum(d[metric_name]
                                               for d in metrics_per_epoch) / len(metrics_per_epoch)
                   for metric_name in metrics_names}
    return metrics_agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training.json",
                        # required=True,
                        help='Path to the json config for training')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    training_params = load_params_from_file(args.config_path)
    logging.info(f"Parameters for training: {training_params}")

    # Wandb
    if training_params['enable_wandb'] is False:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb.init(project="hyperbert", entity="adrianbzgteam")
    wandb.config.update(training_params)

    # Get the model
    model = HyperBertHypergraphModel.from_pretrained(training_params['base_model_name'],
                                                     graph_in_dim=get_graph_input_dim_for_dataset(training_params['dataset_name']))
    # Get the train data loader
    train_dataloader = get_data_loader(training_params)

    trainer = Trainer(params=training_params,
                      model=model,
                      device=get_available_device())

    optimizer = trainer.prepare_optimizers()

    epoch_iterator = trange(training_params["num_epochs"], desc="Training", position=0, leave=True)
    for epoch_idx, epoch in enumerate(epoch_iterator):
        train_log_metrics = run_train_epoch(dataloader=train_dataloader,
                                            trainer=trainer,
                                            optimizer=optimizer)

        wandb.log(train_log_metrics, step=epoch_idx)

        epoch_iterator.set_description(f"Training (Epoch: {epoch_idx + 1} | Loss: {train_log_metrics['train/train_loss']})")
        epoch_iterator.refresh()

    # Close wandb
    wandb.finish()
