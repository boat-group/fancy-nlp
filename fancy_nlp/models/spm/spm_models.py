# -*- coding: utf-8 -*-

from typing import Tuple, List, Union, Optional
import numpy as np

from fancy_nlp.models.spm.base_spm_model import BaseSPMModel
from fancy_nlp.layers.matching import *


class SiameseCNN(BaseSPMModel):
    """Siamese CNN model for SPM.
    """
    def __init__(self,
                 num_class: int,
                 use_word: bool = True,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 use_char: bool = False,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 max_len: Optional[int] = None,
                 max_word_len: Optional[int] = None,
                 dropout: float = 0.2,
                 filters: int = 200,
                 kernel_size: Union[Tuple[int], List[int]] = (2, 3, 4, 5),
                 char_dim: int = 50,
                 fc_dim: int = 200,
                 activation: str = 'relu',
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam'):
        """

        Args:
            num_class: int: the number of classification class
            use_word: boolean, whether to use word embedding as input
            word_embeddings: np.ndarray, word embeddings
            word_vocab_size: int, the number of words in vocabulary
            word_embed_dim: int, dimensionality of word embedding
            word_embed_trainable: boolean, whether to update word embedding during training
            use_char: boolean, whether to use char as input
            char_embeddings: ndarray, char_embeddings
            char_vocab_size: int, the number of chars in vocabulary
            char_embed_dim: int, dimensionality of char embedding
            char_embed_trainable: boolean, similar as 'word_embed_trainable'
            use_bert: boolean, whether to use bert embedding as input
            bert_config_file: str, path to bert's configuration file
            bert_checkpoint_file: str, path to bert's checkpoint file
            bert_trainable: boolean, whether to update bert during training
            max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                     as max_len. However, max_len must be provided when using bert as input.
            max_word_len: int, max word length. If None, we dynamically use the max word length of one
                          batch as max_word_len.
            dropout: float, drop rate
            filters: int, the number of filters for cnn
            kernel_size: list, kernel size for cnn
            char_dim: int, char embedding dim for word+char input
            fc_dim: int, output dimensionality of fully connected layer
            activation: str, activation function name
            optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                       use during training
        """

        self.num_class = num_class
        self.filters = filters
        self.kernel_size = kernel_size
        self.fc_dim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(SiameseCNN, self).__init__(use_word, word_embeddings, word_vocab_size,
                                         word_embed_dim, word_embed_trainable, use_char,
                                         char_embeddings, char_vocab_size, char_embed_dim,
                                         char_embed_trainable, use_bert, bert_config_file,
                                         bert_checkpoint_file, bert_trainable,
                                         False, max_len, max_word_len, char_dim, dropout)

    def build_model(self) -> tf.keras.models.Model:
        model_inputs, input_embed_a, input_embed_b = self.build_input()

        a_conv_layers = []
        b_conv_layers = []
        for filter_length in self.kernel_size:
            conv_layer = tf.keras.layers.Conv1D(filters=self.filters,
                                                kernel_size=filter_length,
                                                padding='valid',
                                                activation='relu',
                                                strides=1)
            a_conv = conv_layer(input_embed_a)
            b_conv = conv_layer(input_embed_b)
            a_conv_layers.append(tf.keras.layers.GlobalMaxPooling1D()(a_conv))
            b_conv_layers.append(tf.keras.layers.GlobalMaxPooling1D()(b_conv))
        a_conv = tf.keras.layers.concatenate(inputs=a_conv_layers)
        b_conv = tf.keras.layers.concatenate(inputs=b_conv_layers)
        sent_rep = tf.keras.layers.concatenate([a_conv, b_conv])
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model


class SiameseBiLSTM(BaseSPMModel):
    """Siamese Bidirectional LSTM model for SPM.
    """
    def __init__(self,
                 num_class: int,
                 use_word: bool = True,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 use_char: bool = False,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 max_len: Optional[int] = None,
                 max_word_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 char_dim: int = 50,
                 fc_dim: int = 200,
                 activation: str = 'relu',
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam'):
        """

        Args:
            num_class: int: the number of classification class
            use_word: boolean, whether to use word embedding as input
            word_embeddings: np.ndarray, word embeddings
            word_vocab_size: int, the number of words in vocabulary
            word_embed_dim: int, dimensionality of word embedding
            word_embed_trainable: boolean, whether to update word embedding during training
            use_char: boolean, whether to use char as input
            char_embeddings: ndarray, char_embeddings
            char_vocab_size: int, the number of chars in vocabulary
            char_embed_dim: int, dimensionality of char embedding
            char_embed_trainable: boolean, similar as 'word_embed_trainable'
            use_bert: boolean, whether to use bert embedding as input
            bert_config_file: str, path to bert's configuration file
            bert_checkpoint_file: str, path to bert's checkpoint file
            bert_trainable: boolean, whether to update bert during training
            max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                     as max_len. However, max_len must be provided when using bert as input.
            max_word_len: int, max word length. If None, we dynamically use the max word length of one
                          batch as max_word_len.
            dropout: float, drop rate
            rnn_units: int, hidden size for lstm
            char_dim: int, char embedding dim for word+char input
            fc_dim: int, output dimensionality of fully connected layer
            activation: str, activation function name
            optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                       use during training
        """

        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_dim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(SiameseBiLSTM, self).__init__(use_word, word_embeddings, word_vocab_size,
                                            word_embed_dim, word_embed_trainable, use_char,
                                            char_embeddings, char_vocab_size, char_embed_dim,
                                            char_embed_trainable, use_bert, bert_config_file,
                                            bert_checkpoint_file, bert_trainable,
                                            False, max_len, max_word_len, char_dim, dropout)

    def build_model(self) -> tf.keras.models.Model:
        model_inputs, input_embed_a, input_embed_b = self.build_input()

        bilstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_units))
        a_lstm = bilstm_layer(input_embed_a)
        b_lstm = bilstm_layer(input_embed_b)

        sent_rep = tf.keras.layers.concatenate([a_lstm, b_lstm])
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model


class SiameseBiGRU(BaseSPMModel):
    """Siamese Bidirectional GRU model for SPM.
    """

    def __init__(self,
                 num_class: int,
                 use_word: bool = True,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 use_char: bool = False,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 max_len: Optional[int] = None,
                 max_word_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 char_dim: int = 50,
                 fc_dim: int = 200,
                 activation: str = 'relu',
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam'):
        """

        Args:
           num_class: int: the number of classification class
           use_word: boolean, whether to use word embedding as input
           word_embeddings: np.ndarray, word embeddings
           word_vocab_size: int, the number of words in vocabulary
           word_embed_dim: int, dimensionality of word embedding
           word_embed_trainable: boolean, whether to update word embedding during training
           use_char: boolean, whether to use char as input
           char_embeddings: ndarray, char_embeddings
           char_vocab_size: int, the number of chars in vocabulary
           char_embed_dim: int, dimensionality of char embedding
           char_embed_trainable: boolean, similar as 'word_embed_trainable'
           use_bert: boolean, whether to use bert embedding as input
           bert_config_file: str, path to bert's configuration file
           bert_checkpoint_file: str, path to bert's checkpoint file
           bert_trainable: boolean, whether to update bert during training
           max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                    as max_len. However, max_len must be provided when using bert as input.
           max_word_len: int, max word length. If None, we dynamically use the max word length of one
                         batch as max_word_len.
           dropout: float, drop rate
           rnn_units: int, hidden size for lstm
           char_dim: int, char embedding dim for word+char input
           fc_dim: int, output dimensionality of fully connected layer
           activation: str, activation function name
           optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                      use during training
        """

        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_dim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(SiameseBiGRU, self).__init__(use_word, word_embeddings, word_vocab_size,
                                           word_embed_dim, word_embed_trainable, use_char,
                                           char_embeddings, char_vocab_size, char_embed_dim,
                                           char_embed_trainable, use_bert, bert_config_file,
                                           bert_checkpoint_file, bert_trainable,
                                           False, max_len, max_word_len, char_dim, dropout)

    def build_model(self) -> tf.keras.models.Model:
        model_inputs, input_embed_a, input_embed_b = self.build_input()

        bigru_layer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.rnn_units))
        a_gru = bigru_layer(input_embed_a)
        b_gru = bigru_layer(input_embed_b)

        sent_rep = tf.keras.layers.concatenate([a_gru, b_gru])
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model


class ESIM(BaseSPMModel):
    """Enhanced LSTM model (esim) for SPM.
    Support using CUDANNLSTM for acceleration when gpu is available.
    """
    def __init__(self,
                 num_class: int,
                 use_word: bool = True,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 use_char: bool = False,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 max_len: Optional[int] = None,
                 max_word_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 char_dim: int = 50,
                 fc_dim: int = 200,
                 activation: str = 'relu',
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam'):
        """

        Args:
           num_class: int: the number of classification class
           use_word: boolean, whether to use word embedding as input
           word_embeddings: np.ndarray, word embeddings
           word_vocab_size: int, the number of words in vocabulary
           word_embed_dim: int, dimensionality of word embedding
           word_embed_trainable: boolean, whether to update word embedding during training
           use_char: boolean, whether to use char as input
           char_embeddings: ndarray, char_embeddings
           char_vocab_size: int, the number of chars in vocabulary
           char_embed_dim: int, dimensionality of char embedding
           char_embed_trainable: boolean, similar as 'word_embed_trainable'
           use_bert: boolean, whether to use bert embedding as input
           bert_config_file: str, path to bert's configuration file
           bert_checkpoint_file: str, path to bert's checkpoint file
           bert_trainable: boolean, whether to update bert during training
           max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                    as max_len. However, max_len must be provided when using bert as input.
           max_word_len: int, max word length. If None, we dynamically use the max word length of one
                         batch as max_word_len.
           dropout: float, drop rate
           rnn_units: int, hidden size for lstm
           char_dim: int, char embedding dim for word+char input
           fc_dim: int, output dimensionality of fully connected layer
           activation: str, activation function name
           optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                      use during training
        """

        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_dim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(ESIM, self).__init__(use_word, word_embeddings, word_vocab_size,
                                   word_embed_dim, word_embed_trainable, use_char,
                                   char_embeddings, char_vocab_size,  char_embed_dim,
                                   char_embed_trainable, use_bert, bert_config_file,
                                   bert_checkpoint_file, bert_trainable,
                                   False, max_len, max_word_len, char_dim, dropout)

    def build_model(self) -> tf.keras.models.Model:
        model_inputs, input_embed_a, input_embed_b = self.build_input()

        bilstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True))
        a_lstm = bilstm_layer(input_embed_a)
        b_lstm = bilstm_layer(input_embed_b)

        # interaction attention
        attention = tf.keras.layers.Dot(axes=-1)([a_lstm, b_lstm])

        wb = tf.keras.layers.Lambda(lambda x: K.softmax(x, axis=1),
                                    output_shape=lambda x: x)(attention)
        wa = tf.keras.layers.Lambda(lambda x: K.softmax(x, axis=2),
                                    output_shape=lambda x: x)(attention)
        wa = tf.keras.layers.Permute((2, 1))(wa)
        a_ = tf.keras.layers.Dot(axes=1)([wa, b_lstm])
        b_ = tf.keras.layers.Dot(axes=1)([wb, a_lstm])

        substract_a = tf.keras.layers.Subtract()([a_lstm, a_])
        multiply_a = tf.keras.layers.Multiply()([a_lstm, a_])
        substract_b = tf.keras.layers.Subtract()([b_lstm, b_])
        multiply_b = tf.keras.layers.Multiply()([b_lstm, b_])

        m_a = tf.keras.layers.concatenate([a_lstm, a_, substract_a, multiply_a], axis=-1)
        m_b = tf.keras.layers.concatenate([b_lstm, b_, substract_b, multiply_b], axis=-1)

        compose = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True))
        v_a = compose(m_a)
        v_b = compose(m_b)

        a_maxpool = tf.keras.layers.GlobalMaxPool1D()(v_a)
        b_maxpool = tf.keras.layers.GlobalMaxPool1D()(v_b)
        a_avgpool = tf.keras.layers.GlobalAvgPool1D()(v_a)
        b_avgpool = tf.keras.layers.GlobalAvgPool1D()(v_b)
        a_pool = tf.keras.layers.concatenate([a_avgpool, a_maxpool], axis=-1)
        b_pool = tf.keras.layers.concatenate([b_avgpool, b_maxpool], axis=-1)

        sent_rep = tf.keras.layers.concatenate([a_pool, b_pool])
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model


class BiMPM(BaseSPMModel):
    """Bilateral Multi-perspective Matching (BiMPM) model for SPM.
    """

    def __init__(self,
                 num_class: int,
                 use_word: bool = True,
                 word_embeddings: Optional[np.ndarray] = None,
                 word_vocab_size: int = -1,
                 word_embed_dim: int = -1,
                 word_embed_trainable: bool = False,
                 use_char: bool = False,
                 char_embeddings: Optional[np.ndarray] = None,
                 char_vocab_size: int = -1,
                 char_embed_dim: int = -1,
                 char_embed_trainable: bool = False,
                 use_bert: bool = False,
                 bert_config_file: Optional[str] = None,
                 bert_checkpoint_file: Optional[str] = None,
                 bert_trainable: bool = False,
                 max_len: Optional[int] = None,
                 max_word_len: Optional[int] = None,
                 dropout: float = 0.2,
                 rnn_units: int = 150,
                 char_dim: int = 50,
                 fc_dim: int = 200,
                 activation: str = 'relu',
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam'):
        """

        Args:
           num_class: int: the number of classification class
           use_word: boolean, whether to use word embedding as input
           word_embeddings: np.ndarray, word embeddings
           word_vocab_size: int, the number of words in vocabulary
           word_embed_dim: int, dimensionality of word embedding
           word_embed_trainable: boolean, whether to update word embedding during training
           use_char: boolean, whether to use char as input
           char_embeddings: ndarray, char_embeddings
           char_vocab_size: int, the number of chars in vocabulary
           char_embed_dim: int, dimensionality of char embedding
           char_embed_trainable: boolean, similar as 'word_embed_trainable'
           use_bert: boolean, whether to use bert embedding as input
           bert_config_file: str, path to bert's configuration file
           bert_checkpoint_file: str, path to bert's checkpoint file
           bert_trainable: boolean, whether to update bert during training
           max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                    as max_len. However, max_len must be provided when using bert as input.
           max_word_len: int, max word length. If None, we dynamically use the max word length of one
                         batch as max_word_len.
           dropout: float, drop rate
           rnn_units: int, hidden size for lstm
           char_dim: int, char embedding dim for word+char input
           fc_dim: int, output dimensionality of fully connected layer
           activation: str, activation function name
           optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                      use during training
        """

        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_dim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(BiMPM, self).__init__(use_word, word_embeddings, word_vocab_size,
                                    word_embed_dim, word_embed_trainable, use_char,
                                    char_embeddings, char_vocab_size, char_embed_dim,
                                    char_embed_trainable, use_bert, bert_config_file,
                                    bert_checkpoint_file, bert_trainable,
                                    False, max_len, max_word_len, char_dim, dropout)

    def build_model(self) -> tf.keras.models.Model:
        model_inputs, input_embed_a, input_embed_b = self.build_input()

        bilstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True))
        a_lstm = bilstm_layer(input_embed_a)
        b_lstm = bilstm_layer(input_embed_b)

        matching1 = FullMatching()
        a_matching1 = matching1([a_lstm, tf.keras.layers.GlobalAvgPool1D()(b_lstm)])
        b_matching1 = matching1([b_lstm, tf.keras.layers.GlobalAvgPool1D()(a_lstm)])

        matching2_forward = MaxPoolingMatching()
        a_matching2 = matching2_forward([a_lstm, b_lstm])
        b_matching2 = matching2_forward([b_lstm, a_lstm])

        cos = tf.keras.layers.Dot(axes=-1, normalize=True)([a_lstm, b_lstm])
        wb = tf.keras.layers.Lambda(lambda x: K.softmax(x, axis=1), output_shape=lambda x: x)(cos)
        wa = tf.keras.layers.Lambda(lambda x: K.softmax(x, axis=2), output_shape=lambda x: x)(cos)
        wa = tf.keras.layers.Permute((2, 1))(wa)

        a_ = tf.keras.layers.Dot(axes=1)([wa, b_lstm])
        b_ = tf.keras.layers.Dot(axes=1)([wb, a_lstm])

        matching3 = AttentiveMatching(perspective_num=5)
        a_matching3 = matching3([a_lstm, a_])
        b_matching3 = matching3([b_lstm, b_])

        a_max = tf.keras.layers.Lambda(
            lambda x: K.batch_dot(K.one_hot(K.argmax(x[0], axis=1), K.int_shape(x[1])[-2]), x[1]),
            output_shape=lambda x: x[1])([wb, a_lstm])
        b_max = tf.keras.layers.Lambda(
            lambda x: K.batch_dot(K.one_hot(K.argmax(x[0], axis=1), K.int_shape(x[1])[-2]), x[1]),
            output_shape=lambda x: x[1])([wa, b_lstm])

        matching4 = MaxAttentiveMatching(perspective_num=5)
        a_matching4 = matching4([a_lstm, b_max])
        b_matching4 = matching4([b_lstm, a_max])

        a_matching = tf.keras.layers.concatenate([a_matching1, a_matching2,
                                                  a_matching3, a_matching4])
        b_matching = tf.keras.layers.concatenate([b_matching1, b_matching2,
                                                  b_matching3, b_matching4])

        aggregation_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_units))
        a_bimpm = aggregation_layer(a_matching)
        b_bimpm = aggregation_layer(b_matching)

        sent_rep = tf.keras.layers.concatenate([a_bimpm, b_bimpm])
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model


class BertSPM(BaseSPMModel):
    """Bert model for SPM.
    We suggest you to train bert on machines with GPU cause it will be very slow to be trained with
    cpu. You will have to re-install a gpu version of tensorflow to do so.
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
                 optimizer: Union[str, tf.keras.optimizers.Optimizer] =
                 tf.keras.optimizers.Adam(lr=1e-5),  # use a small learning rate for bert
                 **kwargs):
        """

        Args:
           num_class: int: the number of classification class
           use_bert: boolean, whether to use bert embedding as input
           bert_config_file: str, path to bert's configuration file
           bert_checkpoint_file: str, path to bert's checkpoint file
           bert_trainable: boolean, whether to update bert during training
           max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                    as max_len. However, max_len must be provided when using bert as input.
           max_word_len: int, max word length. If None, we dynamically use the max word length of one
                         batch as max_word_len.
           dropout: float, drop rate
           fc_dim: int, output dimensionality of fully connected layer
           activation: str, activation function name
           optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                      use during training
        """
        self.num_class = num_class
        self.fc_dim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(BertSPM, self).__init__(use_word=False, use_char=False, use_bert=True,
                                      bert_config_file=bert_config_file,
                                      bert_checkpoint_file=bert_checkpoint_file,
                                      bert_trainable=bert_trainable,
                                      use_bert_model=True,
                                      max_len=max_len, dropout=dropout)

    def build_model(self) -> tf.keras.models.Model:
        model_inputs, input_embed = self.build_input()

        sent_rep = tf.keras.layers.Lambda(lambda x: x[:, 0])(input_embed)
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model
