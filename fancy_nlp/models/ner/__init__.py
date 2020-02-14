# -*- coding: utf-8 -*-

from .ner_models import *

ner_model_dict = {
    'bilstm': BiLSTMNER,
    'bilstm_cnn': BiGRUCNNNER,
    'bigru': BiGRUNER,
    'bigru_cnn': BiGRUCNNNER,
    'bert': BertNER
}
