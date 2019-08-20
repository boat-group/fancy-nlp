# -*- coding: utf-8 -*-

"""Base NER model
"""

from keras.layers import *
from fancy_nlp.models.base_model import BaseModel


class BaseNERModel(BaseModel):
    def __init__(self,
                 checkpoint_dir,
                 model_name,
                 char_embeddings,
                 char_vocab_size,
                 char_embed_dim,
                 char_embed_trainable=False,
                 use_word=False,
                 word_embeddings=None,
                 word_vocab_size=-1,
                 word_embed_dim=-1,
                 word_embed_trainable=False,
                 dropout=0.2,
                 cutsom_objects=None):
        self.char_embeddings = char_embeddings
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.char_embed_trainable = char_embed_trainable
        self.use_word = use_word
        self.word_embeddings = word_embeddings
        self.word_vocab_size = word_vocab_size
        self.word_embed_dim = word_embed_dim
        self.word_embed_trainable = word_embed_trainable
        self.dropout = dropout
        super(BaseNERModel, self).__init__(checkpoint_dir, model_name, cutsom_objects)

    def build_input(self):
        model_inputs = []
        input_embed = []

        # TODO: consider masking
        if self.char_embeddings is not None:
            char_embedding_layer = Embedding(input_dim=self.char_vocab_size,
                                             output_dim=self.char_embed_dim,
                                             weights=[self.char_embeddings],
                                             trainable=self.char_embed_trainable)
        else:
            char_embedding_layer = Embedding(input_dim=self.char_vocab_size,
                                             output_dim=self.char_embed_dim)

        input_char = Input(shape=(None,))
        model_inputs.append(input_char)
        input_embed.append(SpatialDropout1D(self.dropout)(char_embedding_layer(input_char)))

        if self.use_word:
            if self.word_embeddings is not None:
                word_embedding_layer = Embedding(input_dim=self.word_vocab_size,
                                                 output_dim=self.word_embed_dim,
                                                 weights=[self.word_embeddings],
                                                 trainable=self.word_embed_trainable)
            else:
                word_embedding_layer = Embedding(input_dim=self.word_vocab_size,
                                                 output_dim=self.word_embed_dim)
            input_word = Input(shape=(None,))
            model_inputs.append(input_word)
            input_embed.append(SpatialDropout1D(self.dropout)(word_embedding_layer(input_word)))

        input_embed = concatenate(input_embed) if len(input_embed) > 1 else input_embed[0]
        return model_inputs, input_embed

    def build_model_arc(self):
        raise NotImplementedError
