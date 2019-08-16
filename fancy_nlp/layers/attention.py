# -*- coding: utf-8 -*-

from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class MultiHeadAttention(Layer):
    """
    Multi-head Attention introduced in Transformer, support masking
    """
    def __init__(self, num_units=100, num_heads=3, residual=True, normalize=True,
                 initializer='orthogonal', regularizer=None, constraint=None, **kwargs):
        self.num_units = num_units
        self.num_heads = num_heads
        self.model_units = self.num_units * self.num_heads
        self.residual = residual
        self.normalize = normalize
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)
        self.supports_masking = True
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into MultiHeadAttention should be a 3D input tensor')

        self.w_q = self.add_weight(name='w_q', shape=(input_shape[-1], self.model_units),
                                   initializer=self.initializer, regularizer=self.regularizer,
                                   constraint=self.constraint)
        self.w_k = self.add_weight(name='w_k', shape=(input_shape[-1], self.model_units),
                                   initializer=self.initializer, regularizer=self.regularizer,
                                   constraint=self.constraint)
        self.w_v = self.add_weight(name='w_v', shape=(input_shape[-1], self.model_units),
                                   initializer=self.initializer, regularizer=self.regularizer,
                                   constraint=self.constraint)
        self.w_final = self.add_weight(name='w_v', shape=(self.model_units, self.model_units),
                                       initializer=self.initializer, regularizer=self.regularizer,
                                       constraint=self.constraint)
        if self.normalize:
            self.gamma = self.add_weight(name='gamma', shape=(self.model_units,), initializer='one',
                                         regularizer=self.regularizer, constraint=self.constraint)
            self.beta = self.add_weight(name='beta', shape=(self.model_units,), initializer='zero',
                                        regularizer=self.regularizer, constraint=self.constraint)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        convert to query, key, value vectors, shaped [batch_size*num_head, time_step, embed_dim]
        """
        multihead_query = K.concatenate(tf.split(K.dot(inputs, self.w_q),
                                                 self.num_heads, axis=2), axis=0)
        multihead_key = K.concatenate(tf.split(K.dot(inputs, self.w_k),
                                               self.num_heads, axis=2), axis=0)
        multihead_value = K.concatenate(tf.split(K.dot(inputs, self.w_v),
                                                 self.num_heads, axis=2), axis=0)

        """scaled dot product"""
        scaled = K.int_shape(inputs)[-1] ** -0.5
        attend = K.batch_dot(multihead_query, multihead_key, axes=2) * scaled
        # apply mask before normalization (softmax)
        if mask is not None:
            multihead_mask = K.tile(mask, [self.num_heads, 1])
            attend *= K.expand_dims(K.cast(multihead_mask, K.floatx()), 2)
            attend *= K.expand_dims(K.cast(multihead_mask, K.floatx()), 1)
        # normalization
        attend = attend / K.cast(K.sum(attend, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        # apply attention
        attend = K.batch_dot(attend, multihead_value, axes=(2, 1))
        attend = tf.concat(tf.split(attend, self.num_heads, axis=0), axis=2)
        attend = K.dot(attend, self.w_final)

        if self.residual:
            attend = attend + inputs
        if self.normalize:
            mean = K.mean(attend, axis=-1, keepdims=True)
            std = K.mean(attend, axis=-1, keepdims=True)
            attend = self.gamma * (attend - mean) / (std + K.epsilon()) + self.beta

        return attend

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.num_units*self.num_heads

    def get_config(self):
        config = {'num_units': self.num_units,
                  'num_heads': self.num_heads,
                  'residual': self.residual,
                  'normalize': self.normalize,
                  'initializer': initializers.serialize(self.initializer),
                  'regularizer': regularizers.serialize(self.regularizer),
                  'constraint': constraints.serialize(self.constraint)}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
