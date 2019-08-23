# -*- coding: utf-8 -*-

from keras.engine.topology import Layer


class NonMaskingLayer(Layer):
    """
    Fix convolutional 1D can't receive masked input.
    See: https://github.com/keras-team/keras/issues/4978
    Thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x
