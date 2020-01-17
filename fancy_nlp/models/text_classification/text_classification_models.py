# -*- coding: utf-8 -*-

import tensorflow as tf

from fancy_nlp.models.text_classification.base_text_classification_model import \
    BaseTextClassificationModel


class CNNTextClassification(BaseTextClassificationModel):
    """CNN model for text classification.
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
                 optimizer='adam'):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_fim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(CNNTextClassification, self).__init__(
            use_char, char_embeddings, char_vocab_size, char_embed_dim,
            char_embed_trainable, use_bert, bert_config_file,
            bert_checkpoint_file, bert_trainable, use_word,
            word_embeddings, word_vocab_size, word_embed_dim,
            word_embed_trainable, max_len, dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        filter_lengths = [3, 4, 5]
        conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = tf.keras.layers.Conv1D(filters=128,
                                                kernel_size=filter_length,
                                                padding='valid',
                                                activation='relu',
                                                strides=1)(input_embed)
            conv_layers.append(conv_layer)
        poolings = [tf.keras.layers.GlobalMaxPooling1D()(conv) for conv in conv_layers]
        x = tf.keras.layers.Concatenate()(poolings)
        output_layer = tf.keras.layers.Dense(self.num_class, activation='softmax')(x)
        text_classification_loss = 'categorical_crossentropy'
        text_classification_metrics = 'accuracy'
        text_classification_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0], output_layer)
        text_classification_model.compile(
            optimizer=self.optimizer, loss=text_classification_loss,
            metrics=[text_classification_metrics])
        return text_classification_model


class RCNNTextClassification(BaseTextClassificationModel):
    """RCNN model for text classification.
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
                 optimizer='adam'):
        self.num_class = num_class
        self.rnn_units = rnn_units
        self.fc_fim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(RCNNTextClassification, self).__init__(
            use_char, char_embeddings, char_vocab_size, char_embed_dim,
            char_embed_trainable, use_bert, bert_config_file,
            bert_checkpoint_file, bert_trainable, use_word,
            word_embeddings, word_vocab_size, word_embed_dim,
            word_embed_trainable, max_len, dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        input_encode = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True))(input_embed)
        x = tf.keras.layers.Concatenate()([input_embed, input_encode])
        convs = []
        for kernel_size in range(1, 5):
            conv = tf.keras.layers.Conv1D(128, kernel_size, activation='relu')(x)
            convs.append(conv)
        poolings = [tf.keras.layers.GlobalAveragePooling1D()(conv) for conv in convs] + \
                   [tf.keras.layers.GlobalMaxPooling1D()(conv) for conv in convs]
        x = tf.keras.layers.Concatenate()(poolings)
        output_layer = tf.keras.layers.Dense(self.num_class, activation='softmax')(x)

        text_classification_loss = 'categorical_crossentropy'
        text_classification_metrics = 'accuracy'
        text_classification_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0], output_layer)
        text_classification_model.compile(
            optimizer=self.optimizer, loss=text_classification_loss,
            metrics=[text_classification_metrics])
        return text_classification_model


class BertTextClassification(BaseTextClassificationModel):
    """Bert model for text classification.
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
                 optimizer='adam'):
        self.num_class = num_class
        self.fc_fim = fc_dim
        self.activation = activation
        self.optimizer = optimizer
        super(BertTextClassification, self).__init__(
            use_char=False, use_bert=True,
            bert_config_file=bert_config_file,
            bert_checkpoint_file=bert_checkpoint_file,
            bert_trainable=bert_trainable, use_word=False,
            max_len=max_len, dropout=dropout)

    def build_model(self):
        model_inputs, input_embed = self.build_input()
        x = tf.keras.layers.GlobalAveragePooling1D()(input_embed)

        output_layer = tf.keras.layers.Dense(self.num_class, activation='softmax')(x)
        text_classification_loss = 'categorical_crossentropy'
        text_classification_metrics = 'accuracy'
        text_classification_model = tf.keras.models.Model(
            model_inputs if len(model_inputs) > 1 else model_inputs[0], output_layer)
        text_classification_model.compile(
            optimizer=self.optimizer, loss=text_classification_loss,
            metrics=[text_classification_metrics])
        return text_classification_model
