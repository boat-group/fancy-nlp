# -*- coding: utf-8 -*-

import tensorflow as tf

from fancy_nlp.models.ner.base_ner_model import BaseNERModel
from fancy_nlp.layers import CRF
from fancy_nlp.losses import crf_loss
from fancy_nlp.metrics import crf_accuracy


class BiLSTMNER(BaseNERModel):
    """Bidirectional LSTM model for NER.
       Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 use_char=True,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 max_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 fc_dim=100,
                 activation='tanh',
                 use_crf=True,
                 optimizer='adam'):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiLSTMNER, self).__init__(use_char, char_embeddings, char_vocab_size, char_embed_dim,
                                        char_embed_trainable, use_bert, bert_config_file,
                                        bert_checkpoint_file, bert_trainable, use_word,
                                        word_embeddings, word_vocab_size, word_embed_dim,
                                        word_embed_trainable, max_len, dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        input_encode = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0],
            ner_tag
        )
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BiGRUNER(BaseNERModel):
    """Bidirectional GRU model for NER.
       Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 use_char=True,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 max_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 fc_dim=100,
                 activation='tanh',
                 use_crf=True,
                 optimizer='adam'):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiGRUNER, self).__init__(use_char, char_embeddings, char_vocab_size, char_embed_dim,
                                       char_embed_trainable, use_bert, bert_config_file,
                                       bert_checkpoint_file, bert_trainable, use_word,
                                       word_embeddings, word_vocab_size, word_embed_dim,
                                       word_embed_trainable, max_len, dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        input_encode = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0],
            ner_tag
        )
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BiLSTMCNNNER(BaseNERModel):
    """Bidirectional LSTM + CNN model for NER.
       Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 use_char=True,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 max_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 cnn_filters=300,
                 cnn_kernel_size=3,
                 fc_dim=100,
                 activation='tanh',
                 use_crf=True,
                 optimizer='adam'):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiLSTMCNNNER, self).__init__(use_char, char_embeddings, char_vocab_size,
                                           char_embed_dim, char_embed_trainable,
                                           use_bert, bert_config_file, bert_checkpoint_file,
                                           bert_trainable, use_word, word_embeddings,
                                           word_vocab_size, word_embed_dim, word_embed_trainable,
                                           max_len, dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        input_encode = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = tf.keras.layers.Conv1D(filters=self.cnn_filters,
                                              kernel_size=self.cnn_kernel_size,
                                              padding='same',
                                              activation='relu')(input_encode)
        input_encode = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0],
            ner_tag
        )
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BiGRUCNNNER(BaseNERModel):
    """Bidirectional GRU + CNN model for NER.
       Support using CRF layer.
    """
    def __init__(self,
                 num_class,
                 use_char=True,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 max_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 cnn_filters=300,
                 cnn_kernel_size=3,
                 fc_dim=100,
                 activation='tanh',
                 use_crf=True,
                 optimizer='adam'):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BiGRUCNNNER, self).__init__(use_char, char_embeddings, char_vocab_size,
                                          char_embed_dim, char_embed_trainable,
                                          use_bert, bert_config_file, bert_checkpoint_file,
                                          bert_trainable, use_word, word_embeddings,
                                          word_vocab_size, word_embed_dim, word_embed_trainable,
                                          max_len, dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        input_encode = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(self.rnn_units, return_sequences=True))(input_embed)
        input_encode = tf.keras.layers.Conv1D(filters=self.cnn_filters,
                                              kernel_size=self.cnn_kernel_size,
                                              padding='same',
                                              activation='relu')(input_encode)
        input_encode = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.fc_fim, activation=self.activation))(input_encode)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = tf.keras.models.Model(model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model


class BertNER(BaseNERModel):
    """Bert model for NER. Support using CRF layer.
    We suggest you to train bert on machines with GPU cause it will be very slow to be trained with
    cpu. You will have to re-install a gpu version of tensorflow to do so.
    """

    def __init__(self,
                 num_class,
                 bert_config_file,
                 bert_checkpoint_file,
                 bert_trainable,
                 max_len,
                 dropout=0.2,
                 fc_dim=100,
                 activation='tanh',
                 use_crf=True,
                 optimizer=tf.keras.optimizers.Adam(lr=1e-5)):  # use a small learning rate for bert
        self.num_class = num_class
        self.fc_fim = fc_dim
        self.activation = activation
        self.use_crf = use_crf
        self.optimizer = optimizer
        super(BertNER, self).__init__(use_char=False, use_bert=True,
                                      bert_config_file=bert_config_file,
                                      bert_checkpoint_file=bert_checkpoint_file,
                                      bert_trainable=bert_trainable, use_word=False,
                                      max_len=max_len, dropout=dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        input_encode = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.fc_fim, activation=self.activation))(input_embed)

        if self.use_crf:
            crf = CRF(units=self.num_class)
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf_accuracy
        else:
            ner_tag = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_class, activation='softmax'))(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(optimizer=self.optimizer, loss=ner_loss, metrics=[ner_metrics])
        return ner_model
