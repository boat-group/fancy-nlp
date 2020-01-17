# -*- coding: utf-8 -*-

"""Base model with tf.Keras
"""


class BaseModel(object):
    def build_input(self):
        """We build input and embedding layer for tf.keras model here"""
        raise NotImplementedError

    def build_model(self):
        """We build tf.keras model here"""
        raise NotImplementedError
