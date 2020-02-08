# -*- coding: utf-8 -*-
# crf losses that works with tf2.x，originally forked from
# https://github.com/howl-anderson/addons/blob/feature/crf_layers/tensorflow_addons/losses/crf_losses.py

import tensorflow as tf

from ..layers.crf import CRF


def crf_loss(y_true, y_pred):
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
            "Last layer must be CRF for use {}.".format("crf_loss"))

    loss_vector = crf_layer.get_loss(y_true, y_pred)

    return tf.keras.backend.mean(loss_vector)
