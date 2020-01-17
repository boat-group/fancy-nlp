# -*- coding: utf-8 -*-

from keras.layers import *
from keras.models import Model

from fancy_nlp.layers import FullMatching, MaxPoolingMatching, AttentiveMatching, \
    MaxAttentiveMatching


class TestMatching:
    def test_full_matching(self):
        input_embed_a = Input(shape=(3, 300))
        input_embed_b = Input(shape=(300,))
        input_encode = FullMatching()([input_embed_a, input_embed_b])
        model = Model([input_embed_a, input_embed_b], input_encode)

    def test_max_pooling_matching(self):
        input_embed_a = Input(shape=(3, 300))
        input_embed_b = Input(shape=(3, 300))
        input_encode = MaxPoolingMatching()([input_embed_a, input_embed_b])
        model = Model([input_embed_a, input_embed_b], input_encode)

    def test_attentive_matching(self):
        input_embed_a = Input(shape=(3, 300))
        input_embed_b = Input(shape=(3, 300))
        input_encode = AttentiveMatching()([input_embed_a, input_embed_b])
        model = Model([input_embed_a, input_embed_b], input_encode)

    def test_max_attentive_matching(self):
        input_embed_a = Input(shape=(3, 300))
        input_embed_b = Input(shape=(3, 300))
        input_encode = MaxAttentiveMatching()([input_embed_a, input_embed_b])
        model = Model([input_embed_a, input_embed_b], input_encode)