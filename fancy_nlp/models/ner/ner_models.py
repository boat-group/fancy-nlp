# -*- coding: utf-8 -*-

import os

from keras.layers import *
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.models import Model
import tensorflow as tf

from fancy_nlp.models.ner.base_ner_model import BaseNERModel


class BiLSTMNER(BaseNERModel):
    """Bidirectional LSTM model for NER.
    Support using CUDANNLSTM for acceleration when gpu is available.
    Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 checkpoint_dir,
                 char_embeddings,
                 char_vocab_size,
                 char_embed_dim,
                 char_embed_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 dropout=0.5,
                 rnn_units=150,
                 fc_dim=100,
                 activation='relu',
                 use_crf=True,
                 optimizer='adam',
                 model_name=None):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiLSTMNER, self).__init__(checkpoint_dir, model_name if model_name else 'bilstm_ner',
                                        char_embeddings, char_vocab_size, char_embed_dim,
                                        char_embed_trainable, use_word, word_embeddings,
                                        word_vocab_size, word_embed_dim, word_embed_trainable,
                                        dropout,
                                        cutsom_objects={'CRF': CRF} if self.use_crf else None)

    def build_model_arc(self):
        model_inputs, input_embed = self.build_input()
        if tf.test.is_gpu_available(cuda_only=True):
            input_encode = Bidirectional(CuDNNLSTM(self.rnn_units, return_sequences=True))(input_embed)
        else:
            input_encode = Bidirectional(LSTM(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = TimeDistributed(Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = TimeDistributed(Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = Model(model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BiGRUNER(BaseNERModel):
    """Bidirectional GRU model for NER.
        Support using CUDANNGRU for acceleration when gpu is available.
        Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 checkpoint_dir,
                 char_embeddings,
                 char_vocab_size,
                 char_embed_dim,
                 char_embed_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 dropout=0.5,
                 rnn_units=150,
                 fc_dim=100,
                 activation='relu',
                 use_crf=True,
                 optimizer='adam',
                 model_name=None):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiGRUNER, self).__init__(checkpoint_dir, model_name if model_name else 'bigru_ner',
                                       char_embeddings, char_vocab_size, char_embed_dim,
                                       char_embed_trainable, use_word, word_embeddings,
                                       word_vocab_size, word_embed_dim, word_embed_trainable,
                                       dropout, {'CRF': CRF} if use_crf else None)

    def build_model_arc(self):
        model_inputs, input_embed = self.build_input()
        if tf.test.is_gpu_available(cuda_only=True):
            input_encode = Bidirectional(CuDNNGRU(self.rnn_units, return_sequences=True))(input_embed)
        else:
            input_encode = Bidirectional(GRU(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = TimeDistributed(Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = TimeDistributed(Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = Model(model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BiLSTMCNNNER(BaseNERModel):
    """Bidirectional LSTM + CNN model for NER.
        Support using CUDANNLSTM for acceleration when gpu is available.
        Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 checkpoint_dir,
                 char_embeddings,
                 char_vocab_size,
                 char_embed_dim,
                 char_embed_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 dropout=0.5,
                 rnn_units=150,
                 cnn_filters=300,
                 cnn_kernel_size=3,
                 fc_dim=100,
                 activation='relu',
                 use_crf=True,
                 optimizer='adam',
                 model_name=None):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiLSTMCNNNER, self).__init__(checkpoint_dir,
                                           model_name if model_name else 'bilstm_cnn_ner',
                                           char_embeddings, char_vocab_size, char_embed_dim,
                                           char_embed_trainable, use_word, word_embeddings,
                                           word_vocab_size, word_embed_dim, word_embed_trainable,
                                           dropout, {'CRF': CRF} if use_crf else None)

    def build_model_arc(self):
        model_inputs, input_embed = self.build_input()
        if tf.test.is_gpu_available(cuda_only=True):
            input_encode = Bidirectional(CuDNNLSTM(self.rnn_units,
                                                   return_sequences=True))(input_embed)
        else:
            input_encode = Bidirectional(LSTM(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size,
                              padding='same', activation='relu')(input_encode)
        input_encode = TimeDistributed(Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = TimeDistributed(Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = Model(model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BiGRUCNNNER(BaseNERModel):
    """Bidirectional GRU + CNN model for NER.
       Support using CUDANNGRU for acceleration when gpu is available.
       Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 checkpoint_dir,
                 char_embeddings,
                 char_vocab_size,
                 char_embed_dim,
                 char_embed_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 dropout=0.5,
                 rnn_units=150,
                 cnn_filters=300,
                 cnn_kernel_size=3,
                 fc_dim=100,
                 activation='relu',
                 use_crf=True,
                 optimizer='adam',
                 model_name=None):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiGRUCNNNER, self).__init__(checkpoint_dir,
                                          model_name if model_name else 'bigru_cnn_ner',
                                          char_embeddings, char_vocab_size, char_embed_dim,
                                          char_embed_trainable, use_word, word_embeddings,
                                          word_vocab_size, word_embed_dim, word_embed_trainable,
                                          dropout, cutsom_objects={'CRF': CRF} if use_crf else None)

    def build_model_arc(self):
        model_inputs, input_embed = self.build_input()
        if tf.test.is_gpu_available(cuda_only=True):
            input_encode = Bidirectional(CuDNNGRU(self.rnn_units,
                                                  return_sequences=True))(input_embed)
        else:
            input_encode = Bidirectional(GRU(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size,
                              padding='same', activation='relu')(input_encode)
        input_encode = TimeDistributed(Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = TimeDistributed(Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = Model(model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model
