# -*- coding: utf-8 -*-

from absl import logging
import tensorflow as tf


def save_keras_model(model, json_file, weights_file):
    """Save keras model's architecture and weights to disk

    Args:
        model: keras model
        json_file: path to save model's architecture info
        weights_file: path to save model's weights

    Returns:

    """
    model_json = model.to_json()
    with open(json_file, 'w') as writer:
        writer.write(model_json)
    model.save_weights(weights_file)
    logging.info('Saved model to disk')


def load_keras_model(json_file, weights_file, custom_objects=None):
    """Load keras model from disk

    Args:
        json_file: file path to model's architecture info
        weights_file: file path to model's weights
        custom_objects: Optional dictionary mapping names (strings) to custom classes or
                        functions to be considered during deserialization. Must provided when
                        using custom layer
    """
    with open(json_file, 'r') as reader:
        model = tf.keras.models.model_from_json(reader.read(), custom_objects=custom_objects)
    model.load_weights(weights_file)
    return model
