# -*- coding: utf-8 -*-

from keras.layers import *
from keras.models import Model
from fancy_nlp.layers import MultiHeadAttention


class TestAttention:
    def test_multihead_attention(self):
        input_embed = Input(shape=(3, 300))
        input_encode = MultiHeadAttention()(input_embed)
        model = Model(input_embed, input_encode)
