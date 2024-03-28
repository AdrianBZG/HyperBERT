import pytest
import torch

from models.hyperbert_base import HyperBertBaseModel
from models.hyperbert_graph import HyperBertGraphModel
from utils import set_seed, get_huggingface_model, get_dummy_graph_dataset


def test_get_huggingface_model():
    model_dict = get_huggingface_model('bert-base-uncased')
    assert 'tokenizer' in model_dict
    assert 'model' in model_dict


def test_pretrained_bert_equivalence():
    base_model = "bert-base-uncased"
    text = "This is a dummy text."
    set_seed()

    # Get BERT base from huggingface
    model_dict = get_huggingface_model(base_model)
    tokenizer = model_dict['tokenizer']
    bert_base = model_dict['model']

    # Get HyperBert base from our model
    hyperbert_base = HyperBertBaseModel.from_pretrained(base_model)

    # Encode input and get cls embeddings
    encoded_input = tokenizer(text, return_tensors='pt')
    bert_cls_embeddings = bert_base(**encoded_input, return_dict=True).pooler_output
    hyperbert_cls_embeddings = hyperbert_base(**encoded_input, return_dict=True).pooler_output

    assert torch.allclose(bert_cls_embeddings, hyperbert_cls_embeddings, atol=1e-5)


def test_hyperbert_graph():
    base_model = "bert-base-uncased"
    text = "This is a dummy text."
    set_seed()

    # Get BERT base from huggingface
    model_dict = get_huggingface_model(base_model)
    tokenizer = model_dict['tokenizer']
    encoded_input = tokenizer(text, return_tensors='pt')

    dummy_graph = get_dummy_graph_dataset()[0]
    dummy_graph_batch = [dummy_graph]

    # Get HyperBertGraph
    hyperbert_graph = HyperBertGraphModel.from_pretrained(base_model)

    hyperbert_graph_cls_embeddings = hyperbert_graph(**encoded_input,
                                                     graph_batch=dummy_graph_batch,
                                                     return_dict=True).pooler_output

    assert isinstance(hyperbert_graph_cls_embeddings, torch.FloatTensor)
