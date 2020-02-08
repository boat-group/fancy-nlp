# -*- coding: utf-8 -*-

"""Base SPM model
"""

import tensorflow as tf
from bert4keras.bert import build_bert_model

from fancy_nlp.layers import NonMaskingLayer
from fancy_nlp.models.base_model import BaseModel


class BaseSPMModel(BaseModel):
    def __init__(self,
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
                 use_bert_model=False,
                 max_len=None,
                 max_word_len=None,
                 char_dim=50,
                 dropout=0.2):

        self.use_word = use_word
        self.word_embeddings = word_embeddings
        self.word_vocab_size = word_vocab_size
        self.word_embed_dim = word_embed_dim
        self.word_embed_trainable = word_embed_trainable
        self.use_char = use_char
        self.char_embeddings = char_embeddings
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.char_embed_trainable = char_embed_trainable
        self.use_bert = use_bert
        self.bert_config_file = bert_config_file
        self.bert_checkpoint_file = bert_checkpoint_file
        self.bert_trainable = bert_trainable
        self.use_bert_model = use_bert_model
        self.max_len = max_len
        self.max_word_len = max_word_len
        self.char_dim = char_dim
        self.dropout = dropout

        assert not (self.use_bert_model and (self.use_word or self.use_char)), \
            "bert model can not add word or char embedding as additional input"
        assert not (self.use_bert_model and not use_bert), "bert model must use bert" \
                                                           "embedding"
        assert self.use_word or self.use_char or self.use_bert, \
            "must use word or char or bert embedding as input"
        assert not (self.use_word and self.use_bert), \
            "bert embedding can not be used with word embedding"
        assert not (self.use_bert and self.max_len is None), \
            "max_len must be provided when using bert embedding as input"

    def build_char_embedding(self, char_embedding_layer, input_char_a, input_char_b):
        char_embedding_a = tf.keras.layers.TimeDistributed(char_embedding_layer)(input_char_a)
        char_embedding_b = tf.keras.layers.TimeDistributed(char_embedding_layer)(input_char_b)
        conv_layer = tf.keras.layers.Conv1D(filters=self.char_dim, kernel_size=2, padding='same',
                                            activation='relu', strides=1)
        global_maxpool_layer = tf.keras.layers.GlobalMaxPooling1D()
        char_embedding_a = tf.keras.layers.TimeDistributed(conv_layer)(char_embedding_a)
        char_embedding_a = tf.keras.layers.TimeDistributed(global_maxpool_layer)(char_embedding_a)
        char_embedding_b = tf.keras.layers.TimeDistributed(conv_layer)(char_embedding_b)
        char_embedding_b = tf.keras.layers.TimeDistributed(global_maxpool_layer)(char_embedding_b)
        return char_embedding_a, char_embedding_b

    def build_input(self):

        # TODO: consider masking
        if self.use_bert_model:
            model_inputs = []
            bert_model = build_bert_model(config_path=self.bert_config_file,
                                          checkpoint_path=self.bert_checkpoint_file)
            if not self.bert_trainable:
                # manually set every layer in bert model to be non-trainable
                for layer in bert_model.layers:
                    layer.trainable = False

            input_bert = tf.keras.layers.Input(shape=(self.max_len,))
            input_seg = tf.keras.layers.Input(shape=(self.max_len,))
            model_inputs.append(input_bert)
            model_inputs.append(input_seg)
            bert_embed = NonMaskingLayer()(bert_model([input_bert, input_seg]))
            input_embed = tf.keras.layers.SpatialDropout1D(self.dropout)(bert_embed)

            return model_inputs, input_embed

        model_inputs_a = []
        input_embed_a = []
        model_inputs_b = []
        input_embed_b = []

        if self.use_word:
            if self.word_embeddings is not None:
                word_embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.word_vocab_size, output_dim=self.word_embed_dim,
                    weights=[self.word_embeddings], trainable=self.word_embed_trainable)
            else:
                word_embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.word_vocab_size, output_dim=self.word_embed_dim)

            input_word_a = tf.keras.layers.Input(shape=(self.max_len,))
            model_inputs_a.append(input_word_a)
            input_embed_a.append(tf.keras.layers.SpatialDropout1D(self.dropout)(
                word_embedding_layer(input_word_a)))
            input_word_b = tf.keras.layers.Input(shape=(self.max_len,))
            model_inputs_b.append(input_word_b)
            input_embed_b.append(tf.keras.layers.SpatialDropout1D(self.dropout)(
                word_embedding_layer(input_word_b)))

            if self.use_char:
                if self.char_embeddings is not None:
                    char_embedding_layer = tf.keras.layers.Embedding(
                        input_dim=self.char_vocab_size, output_dim=self.char_embed_dim,
                        weights=[self.char_embeddings], trainable=self.char_embed_trainable)
                else:
                    char_embedding_layer = tf.keras.layers.Embedding(
                        input_dim=self.char_vocab_size, output_dim=self.char_embed_dim)

                input_char_a = tf.keras.layers.Input(shape=(self.max_len, self.max_word_len))
                model_inputs_a.append(input_char_a)
                input_char_b = tf.keras.layers.Input(shape=(self.max_len, self.max_word_len))
                model_inputs_b.append(input_char_b)
                char_embed_a, char_embed_b = self.build_char_embedding(char_embedding_layer,
                                                                       input_char_a,
                                                                       input_char_b)
                input_embed_a.append(tf.keras.layers.SpatialDropout1D(self.dropout)(char_embed_a))
                input_embed_b.append(tf.keras.layers.SpatialDropout1D(self.dropout)(char_embed_b))

        else:
            if self.use_char:
                if self.char_embeddings is not None:
                    char_embedding_layer = tf.keras.layers.Embedding(
                        input_dim=self.char_vocab_size, output_dim=self.char_embed_dim,
                        weights=[self.char_embeddings], trainable=self.char_embed_trainable)
                else:
                    char_embedding_layer = tf.keras.layers.Embedding(
                        input_dim=self.char_vocab_size, output_dim=self.char_embed_dim)

                input_char_a = tf.keras.layers.Input(shape=(self.max_len,))
                model_inputs_a.append(input_char_a)
                input_embed_a.append(tf.keras.layers.SpatialDropout1D(self.dropout)(
                    char_embedding_layer(input_char_a)))
                input_char_b = tf.keras.layers.Input(shape=(self.max_len,))
                model_inputs_b.append(input_char_b)
                input_embed_b.append(tf.keras.layers.SpatialDropout1D(self.dropout)(
                    char_embedding_layer(input_char_b)))

            if self.use_bert:
                bert_model = build_bert_model(config_path=self.bert_config_file,
                                              checkpoint_path=self.bert_checkpoint_file)
                if not self.bert_trainable:
                    # manually set every layer in bert model to be non-trainable
                    for layer in bert_model.layers:
                        layer.trainable = False

                input_bert_a = tf.keras.layers.Input(shape=(self.max_len,))
                input_seg_a = tf.keras.layers.Input(shape=(self.max_len,))
                model_inputs_a.append(input_bert_a)
                model_inputs_a.append(input_seg_a)
                bert_embed_a = NonMaskingLayer()(bert_model([input_bert_a, input_seg_a]))
                input_embed_a.append(tf.keras.layers.SpatialDropout1D(self.dropout)(bert_embed_a))

                input_bert_b = tf.keras.layers.Input(shape=(self.max_len,))
                input_seg_b = tf.keras.layers.Input(shape=(self.max_len,))
                model_inputs_b.append(input_bert_b)
                model_inputs_b.append(input_seg_b)
                bert_embed_b = NonMaskingLayer()(bert_model([input_bert_b, input_seg_b]))
                input_embed_b.append(tf.keras.layers.SpatialDropout1D(self.dropout)(bert_embed_b))

        input_embed_a = tf.keras.layers.concatenate(input_embed_a) if len(input_embed_a) > 1 \
            else input_embed_a[0]
        input_embed_b = tf.keras.layers.concatenate(input_embed_b) if len(input_embed_b) > 1 \
            else input_embed_b[0]
        return model_inputs_a + model_inputs_b, input_embed_a, input_embed_b

    def build_model(self):
        raise NotImplementedError
