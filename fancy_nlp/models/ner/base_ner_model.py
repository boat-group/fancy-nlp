# -*- coding: utf-8 -*-

"""Base NER model
"""

from typing import Optional

import numpy as np
import tensorflow as tf
from bert4keras.bert import build_bert_model

from fancy_nlp.layers import NonMaskingLayer
from fancy_nlp.models.base_model import BaseModel


class BaseNERModel(BaseModel):
    """The basic class for ner models. All the ner models will inherit from it.

    """
    def __init__(self,
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
                 dropout: float = 0.2) -> None:
        """

        Args:
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
        """

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
        """Build input placeholder and prepare embedding for ner model.

        Returns: Tuples of 2 tensor:
            1). Input tensor(s), depending whether using multiple inputs;
            2). Embedding tensor, which will be passed to following layers of ner models.

        """
        model_inputs = []
        input_embed = []

        # TODO: consider masking
        if self.use_char:
            if self.char_embeddings is not None:
                char_embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.char_vocab_size,
                    output_dim=self.char_embed_dim,
                    weights=[self.char_embeddings],
                    trainable=self.char_embed_trainable)
            else:
                char_embedding_layer = tf.keras.layers.Embedding(input_dim=self.char_vocab_size,
                                                                 output_dim=self.char_embed_dim)
            input_char = tf.keras.layers.Input(shape=(self.max_len,))
            model_inputs.append(input_char)

            char_embed = char_embedding_layer(input_char)
            input_embed.append(tf.keras.layers.SpatialDropout1D(self.dropout)(char_embed))

        if self.use_bert:
            bert_model = build_bert_model(config_path=self.bert_config_file,
                                          checkpoint_path=self.bert_checkpoint_file)
            if not self.bert_trainable:
                # manually set every layer in bert model to be non-trainable
                for layer in bert_model.layers:
                    layer.trainable = False

            model_inputs.extend(bert_model.inputs)
            bert_embed = NonMaskingLayer()(bert_model.output)
            input_embed.append(tf.keras.layers.SpatialDropout1D(0.2)(bert_embed))

        if self.use_word:
            if self.word_embeddings is not None:
                word_embedding_layer = tf.keras.layers.Embedding(
                    input_dim=self.word_vocab_size,
                    output_dim=self.word_embed_dim,
                    weights=[self.word_embeddings],
                    trainable=self.word_embed_trainable)
            else:
                word_embedding_layer = tf.keras.layers.Embedding(input_dim=self.word_vocab_size,
                                                                 output_dim=self.word_embed_dim)
            input_word = tf.keras.layers.Input(shape=(self.max_len,))
            model_inputs.append(input_word)

            word_embed = word_embedding_layer(input_word)
            input_embed.append(tf.keras.layers.SpatialDropout1D(self.dropout)(word_embed))

        if len(input_embed) > 1:
            input_embed = tf.keras.layers.concatenate(input_embed)
        else:
            input_embed = input_embed[0]
        return model_inputs, input_embed

    def build_model(self):
        """Build ner model's architecture."""
        raise NotImplementedError
