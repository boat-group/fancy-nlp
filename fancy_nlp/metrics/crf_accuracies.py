# -*- coding: utf-8 -*-
# crf metrics that works with tf2.x


import tensorflow as tf

from ..layers.crf import CRF


def crf_accuracy(y_true, y_pred):
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.
    Returns:
        scalar.
    """
    crf_layer = y_pred._keras_history[0]

    # check if last layer is CRF
    if not isinstance(crf_layer, CRF):
        raise ValueError(
            "Last layer must be CRF for use {}.".format("crf_accuracy"))

    accuracy = crf_layer.get_accuracy(y_true, y_pred)

    return accuracy
