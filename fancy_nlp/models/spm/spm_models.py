# -*- coding: utf-8 -*-

import tensorflow as tf

from fancy_nlp.models.spm.base_spm_model import BaseSPMModel
from fancy_nlp.layers.matching import *


class SiameseCNN(BaseSPMModel):
    """Siamese CNN model for SPM.
    """
    def __init__(self,
                 num_class,
                 use_word=True,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 use_char=False,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 max_len=None,
                 max_word_len=None,
                 dropout=0.2,
                 filters=200,
                 kernel_size=(2, 3, 4, 5),
                 char_dim=50,
                 fc_dim=200,
                 activation='relu',
                 optimizer='adam'):
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

    def build_model(self):
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
                 num_class,
                 use_word=True,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 use_char=False,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 max_len=None,
                 max_word_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 char_dim=50,
                 fc_dim=200,
                 activation='relu',
                 optimizer='adam'):
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

    def build_model(self):
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
                 num_class,
                 use_word=True,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 use_char=False,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 bert_output_layer_num=1,
                 max_len=None,
                 max_word_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 char_dim=50,
                 fc_dim=200,
                 activation='relu',
                 optimizer='adam'):
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

    def build_model(self):
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
                 num_class,
                 use_word=True,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 use_char=False,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 bert_output_layer_num=1,
                 max_len=None,
                 max_word_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 char_dim=50,
                 fc_dim=200,
                 activation='relu',
                 optimizer='adam'):
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

    def build_model(self):
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
                 num_class,
                 use_word=True,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 use_char=False,
                 char_embeddings=None,
                 char_vocab_size=-1,
                 char_embed_dim=-1,
                 char_embed_trainable=False,
                 use_bert=False,
                 bert_config_file=None,
                 bert_checkpoint_file=None,
                 bert_trainable=False,
                 bert_output_layer_num=1,
                 max_len=None,
                 max_word_len=None,
                 dropout=0.2,
                 rnn_units=150,
                 char_dim=50,
                 fc_dim=200,
                 activation='relu',
                 optimizer='adam'):
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

    def build_model(self):
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
                 num_class,
                 bert_config_file,
                 bert_checkpoint_file,
                 bert_trainable,
                 max_len,
                 dropout=0.2,
                 fc_dim=100,
                 activation='tanh',
                 optimizer=tf.keras.optimizers.Adam(lr=1e-5),  # use a small learning rate for bert
                 bert_output_layer_num=1,
                 **kwargs):
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

    def build_model(self):
        model_inputs, input_embed = self.build_input()

        sent_rep = tf.keras.layers.Lambda(lambda x: x[:, 0])(input_embed)
        sent_rep = tf.keras.layers.Dense(self.fc_dim, activation=self.activation)(sent_rep)
        matching_score = tf.keras.layers.Dense(self.num_class, activation='softmax')(sent_rep)

        spm_loss = 'categorical_crossentropy'
        spm_metrics = 'accuracy'
        spm_model = tf.keras.models.Model(model_inputs, matching_score)
        spm_model.compile(optimizer=self.optimizer, loss=spm_loss, metrics=[spm_metrics])
        return spm_model
