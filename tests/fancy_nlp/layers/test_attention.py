# -*- coding: utf-8 -*-

import tensorflow as tf

from fancy_nlp.layers import MultiHeadAttention


class TestAttention:
    def test_multihead_attention(self):
        input_embed = tf.keras.layers.Input(shape=(3, 300))
        input_encode = MultiHeadAttention()(input_embed)
        model = tf.keras.models.Model(input_embed, input_encode)
