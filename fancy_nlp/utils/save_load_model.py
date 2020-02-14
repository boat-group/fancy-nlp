# -*- coding: utf-8 -*-

from typing import Dict, Any

from absl import logging
import tensorflow as tf


def save_keras_model(model: tf.keras.models.Model, json_file: str, weights_file: str) -> None:
    """Save keras model's architecture and weights to disk

    Args:
        model: Instance of tf.keras model.
        json_file: str, Path to save model's architecture info.
        weights_file: str. Path to save model's weights.

    Returns:

    """
    model_json = model.to_json()
    with open(json_file, 'w') as writer:
        writer.write(model_json)
    model.save_weights(weights_file)
    logging.info('Saved model to disk')


def load_keras_model(json_file: str,
                     weights_file: str,
                     custom_objects: Dict[str, Any] = None) -> tf.keras.models.Model:
    """Load keras model from disk

    Args:
        json_file: str. File path to model's architecture info.
        weights_file: str. File path to model's weights.
        custom_objects: Optional dictionary mapping names (strings) to custom classes or
                        functions to be considered during deserialization. Must provided when
                        using custom layer.
    """
    with open(json_file, 'r') as reader:
        model = tf.keras.models.model_from_json(reader.read(), custom_objects=custom_objects)
    model.load_weights(weights_file)
    return model
