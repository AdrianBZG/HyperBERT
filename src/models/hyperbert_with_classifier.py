import logging
import torch.nn as nn

from models.hyperbert_graph import HyperBertGraphModel
from models.hyperbert_hypergraph import HyperBertHypergraphModel
from utils import freeze_model

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


class HyperBertWithClassifier(nn.Module):
    def __init__(self, encoder, classifier, freeze_encoder=True):
        """
        General module which contains (i) a pre-trained HyperBert and (ii) a classification head for fine-tuning.

        Parameters:
        - encoder: A hyperbert model that can encode a node text and graph structure
        - classifier: A model that implements .forward() for classification
        """
        super().__init__()

        if not isinstance(encoder, HyperBertHypergraphModel) and not isinstance(encoder, HyperBertGraphModel):
            raise ValueError(f'Unsupported encoder type {type(encoder)}')

        if classifier and not isinstance(classifier, nn.Module):
            raise ValueError(f'Unsupported classifier type {type(classifier)}. '
                             f'It has to be a subclass of torch.nn.Module')

        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            freeze_model(self.encoder)

    def encode(self, kwargs):
        return self.encoder(**kwargs)

    def forward(self, kwargs):
        x_encoded = self.encoder(**kwargs)
        return self.classifier(x_encoded, as_probabilities=False)
