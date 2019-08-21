# -*- coding: utf-8 -*-

"""Base NER model
"""

from keras.layers import *
from keras_bert import load_trained_model_from_checkpoint

from fancy_nlp.models.base_model import BaseModel


class BaseNERModel(BaseModel):
    def __init__(self,
                 checkpoint_dir,
                 model_name,
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
                 custom_objects=None):
        super(BaseNERModel, self).__init__(checkpoint_dir, model_name, custom_objects)

        self.use_char = use_char
        self.char_embeddings = char_embeddings
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.char_embed_trainable = char_embed_trainable
        self.use_bert = use_bert
        self.bert_config_file = bert_config_file
        self.bert_checkpoint_file = bert_checkpoint_file
        self.bert_trainable = bert_trainable
        self.use_word = use_word
        self.word_embeddings = word_embeddings
        self.word_vocab_size = word_vocab_size
        self.word_embed_dim = word_embed_dim
        self.word_embed_trainable = word_embed_trainable
        self.max_len = max_len
        self.dropout = dropout

        assert self.use_char or self.use_bert, "must use char or bert embedding as main input"
        assert not (self.use_bert and self.max_len is None), \
            "max_len must be provided when using bert embedding as input"

    def build_input(self):
        model_inputs = []
        input_embed = []

        # TODO: consider masking
        if self.use_char:
            if self.char_embeddings is not None:
                char_embedding_layer = Embedding(input_dim=self.char_vocab_size,
                                                 output_dim=self.char_embed_dim,
                                                 weights=[self.char_embeddings],
                                                 trainable=self.char_embed_trainable)
            else:
                char_embedding_layer = Embedding(input_dim=self.char_vocab_size,
                                                 output_dim=self.char_embed_dim)
            input_char = Input(shape=(self.max_len,))
            model_inputs.append(input_char)
            input_embed.append(SpatialDropout1D(self.dropout)(char_embedding_layer(input_char)))

        if self.use_bert:
            bert_model = load_trained_model_from_checkpoint(self.bert_config_file,
                                                            self.bert_checkpoint_file,
                                                            self.bert_trainable,
                                                            self.max_len)
            model_inputs.append(bert_model.inputs)
            input_embed.append(SpatialDropout1D(0.2)(bert_model.output))

        if self.use_word:
            if self.word_embeddings is not None:
                word_embedding_layer = Embedding(input_dim=self.word_vocab_size,
                                                 output_dim=self.word_embed_dim,
                                                 weights=[self.word_embeddings],
                                                 trainable=self.word_embed_trainable)
            else:
                word_embedding_layer = Embedding(input_dim=self.word_vocab_size,
                                                 output_dim=self.word_embed_dim)
            input_word = Input(shape=(self.max_len,))
            model_inputs.append(input_word)
            input_embed.append(SpatialDropout1D(self.dropout)(word_embedding_layer(input_word)))

        input_embed = concatenate(input_embed) if len(input_embed) > 1 else input_embed[0]
        return model_inputs, input_embed

    def build_model_arc(self):
        raise NotImplementedError
