# -*- coding: utf-8 -*-

from typing import Optional, Union

import tensorflow as tf
import numpy as np

from fancy_nlp.models.ner.base_ner_model import BaseNERModel
from fancy_nlp.layers import CRF
from fancy_nlp.losses import crf_loss
from fancy_nlp.metrics import crf_accuracy


class BiLSTMNER(BaseNERModel):
    """Bidirectional LSTM model for NER.
       Support using CRF layer.
    """
    def __init__(self,
                 num_class: int,
                 use_char: bool = True,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 use_word: bool = False,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 max_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 fc_dim: int = 100,
                 activation: str = 'tanh',
                 use_crf: bool = True,
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                 **kwargs):
        """

        Args:
            num_class: int. Number of entity type.
            use_char: Boolean. Whether to use character embedding as input.
            char_embeddings: Optional np.ndarray. Char embedding matrix, shaped
                [char_vocab_size, char_embed_dim]. There are 2 cases when char_embeddings is None:
                1)  use_char is False, do not use char embedding as input; 2) user did not
                provide valid pre-trained embedding file or any embedding training method. In
                this case, use randomly initialized embedding instead.
            char_vocab_size: int. The size of char vocabulary.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            word_embeddings: Optional np.ndarray. Similar as char_embeddings.
            word_vocab_size: int. Similar as char_vocab_size.
            word_embed_dim: int. Similar as char_embed_dim.
            word_embed_trainable: Boolean. Similar as char_embed_trainable.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
            rnn_units: int. Dimensionality of the LSTM units.
            fc_dim: int. Dimensionality of fully-connected layer.
            activation: str. Activation function to use in fully-connected layer.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs:
        """
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

    def build_model(self) -> tf.keras.models.Model:
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
                 num_class: int,
                 use_char: bool = True,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 use_word: bool = False,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 max_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 fc_dim: int = 100,
                 activation: str = 'tanh',
                 use_crf: bool = True,
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                 **kwargs):
        """

        Args:
            num_class: int. Number of entity type.
            use_char: Boolean. Whether to use character embedding as input.
            char_embeddings: Optional np.ndarray. Char embedding matrix, shaped
                [char_vocab_size, char_embed_dim]. There are 2 cases when char_embeddings is None:
                1)  use_char is False, do not use char embedding as input; 2) user did not
                provide valid pre-trained embedding file or any embedding training method. In
                this case, use randomly initialized embedding instead.
            char_vocab_size: int. The size of char vocabulary.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            word_embeddings: Optional np.ndarray. Similar as char_embeddings.
            word_vocab_size: int. Similar as char_vocab_size.
            word_embed_dim: int. Similar as char_embed_dim.
            word_embed_trainable: Boolean. Similar as char_embed_trainable.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
            rnn_units: int. Dimensionality of the LSTM units.
            fc_dim: int. Dimensionality of fully-connected layer.
            activation: str. Activation function to use in fully-connected layer.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs:
        """
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

    def build_model(self) -> tf.keras.Model:
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
                 num_class: int,
                 use_char: bool = True,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 use_word: bool = False,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 max_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 cnn_filters: int = 300,
                 cnn_kernel_size: int = 3,
                 fc_dim: int = 100,
                 activation: str = 'tanh',
                 use_crf: bool = True,
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                 **kwargs):
        """

        Args:
            num_class: int. Number of entity type.
            use_char: Boolean. Whether to use character embedding as input.
            char_embeddings: Optional np.ndarray. Char embedding matrix, shaped
                [char_vocab_size, char_embed_dim]. There are 2 cases when char_embeddings is None:
                1)  use_char is False, do not use char embedding as input; 2) user did not
                provide valid pre-trained embedding file or any embedding training method. In
                this case, use randomly initialized embedding instead.
            char_vocab_size: int. The size of char vocabulary.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            word_embeddings: Optional np.ndarray. Similar as char_embeddings.
            word_vocab_size: int. Similar as char_vocab_size.
            word_embed_dim: int. Similar as char_embed_dim.
            word_embed_trainable: Boolean. Similar as char_embed_trainable.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
            rnn_units: int. Dimensionality of the LSTM units.
            cnn_filters: int. The number of output filters in the convolution
            cnn_kernel_size: int. The length of the 1D convolution window.
            fc_dim: int. Dimensionality of fully-connected layer.
            activation: str. Activation function to use in fully-connected layer.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs:
        """
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

    def build_model(self) -> tf.keras.Model:
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
                 num_class: int,
                 use_char: bool = True,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 use_word: bool = False,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 max_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 cnn_filters: int = 300,
                 cnn_kernel_size: int = 3,
                 fc_dim: int = 100,
                 activation: str = 'tanh',
                 use_crf: bool = True,
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                 **kwargs):
        """

        Args:
            num_class: int. Number of entity type.
            use_char: Boolean. Whether to use character embedding as input.
            char_embeddings: Optional np.ndarray. Char embedding matrix, shaped
                [char_vocab_size, char_embed_dim]. There are 2 cases when char_embeddings is None:
                1)  use_char is False, do not use char embedding as input; 2) user did not
                provide valid pre-trained embedding file or any embedding training method. In
                this case, use randomly initialized embedding instead.
            char_vocab_size: int. The size of char vocabulary.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            word_embeddings: Optional np.ndarray. Similar as char_embeddings.
            word_vocab_size: int. Similar as char_vocab_size.
            word_embed_dim: int. Similar as char_embed_dim.
            word_embed_trainable: Boolean. Similar as char_embed_trainable.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
            rnn_units: int. Dimensionality of the LSTM units.
            cnn_filters: int. The number of output filters in the convolution
            cnn_kernel_size: int. The length of the 1D convolution window.
            fc_dim: int. Dimensionality of fully-connected layer.
            activation: str. Activation function to use in fully-connected layer.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs:
        """
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

    def build_model(self) -> tf.keras.models.Model:
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
    cpu.
    """

    def __init__(self,
                 num_class: int,
                 bert_config_file: str,
                 bert_checkpoint_file: str,
                 bert_trainable: bool,
                 max_len: int,
                 dropout: float = 0.2,
                 fc_dim: int = 100,
                 activation: str = 'tanh',
                 use_crf: bool = True,
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = tf.keras.optimizers.Adam(
                     lr=1e-5),
                 # use a small
                 # learning
                 # rate for
                 **kwargs):
        """

        Args:
            num_class: int. Number of entity type.
            bert_config_file: str. Path to bert's configuration file.
            bert_checkpoint_file: str. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
            fc_dim: int. Dimensionality of fully-connected layer.
            activation: str. Activation function to use in fully-connected layer.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs:
        """
        # bert
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

    def build_model(self) -> tf.keras.models.Model:
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
