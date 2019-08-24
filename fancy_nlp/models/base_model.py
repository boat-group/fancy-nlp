# -*- coding: utf-8 -*-

"""Base model with Keras
"""


class BaseModel(object):
    def build_input(self):
        """We build input and embedding layer for keras model here"""
        raise NotImplementedError

    def build_model(self):
        """We build keras model here"""
        raise NotImplementedError
