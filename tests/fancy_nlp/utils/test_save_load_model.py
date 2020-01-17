# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import tensorflow.keras.backend as K

from fancy_nlp.utils import save_keras_model, load_keras_model


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        return K.dot(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {'output_dim': self.output_dim}

        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TestSaveLoad:
    test_json_file = 'my_model_architecture.json'
    test_weights_file = 'my_model_weights.h5'

    def setup_class(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(32, input_shape=(784, )))
        self.model.add(MyLayer(100, name='my_layer1'))
        self.model.add(MyLayer(100, name='my_layer2'))

    def test_save(self):
        save_keras_model(self.model, self.test_json_file, self.test_weights_file)

    def test_load(self):
        self.model = load_keras_model(self.test_json_file, self.test_weights_file,
                                      custom_objects={'MyLayer': MyLayer})
        assert len(self.model.layers) == 3

    def teardown_class(self):
        os.remove(self.test_json_file)
        os.remove(self.test_weights_file)
