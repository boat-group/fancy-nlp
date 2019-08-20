# -*- coding: utf-8 -*-

"""Base model with Keras
"""

import os

from absl import logging
from keras.models import model_from_json


class BaseModel(object):
    def __init__(self,
                 checkpoint_dir,
                 model_name,
                 custom_objects=None):
        """

        Args:
            checkpoint_dir: directory to save the model
            model_name: model name
            custom_objects: Optional dictionary mapping names (strings) to custom classes or
                            functions to be considered during deserialization.
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.json_file = os.path.join(self.checkpoint_dir, self.model_name+'.json')
        self.weights_file = os.path.join(self.checkpoint_dir, self.model_name+'.hdf5')
        self.swa_weights_file = os.path.join(self.checkpoint_dir, self.model_name+'_swa.hdf5')
        self.custom_objects = custom_objects

        self.model = None

    def build_input(self):
        """We build input and embedding layer for keras model here"""
        raise NotImplementedError

    def build_model_arc(self):
        """We build keras model here"""
        raise NotImplementedError

    def build_model(self):
        if self.model is None:
            self.model = self.build_model_arc()

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, batch_size=32, epochs=50,
            callbacks=None):
        """Train model

        Args:
            x_train: list of training data
            y_train: list of training labels
            x_valid: list of validation data
            y_valid: list of validation labels
            batch_size: num of samples per gradient update
            epochs: num of epochs to train the model
            callbacks: List of `keras.callbacks.Callback` instances to apply during training.

        """
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')

        logging.info('Training start...')
        valid_data = (x_valid, y_valid) if x_valid is not None and y_valid is not None else None
        self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=valid_data, callbacks=callbacks)
        logging.info('Training end...')

    def fit_generator(self, train_generator, valid_generator=None, epochs=50, callbacks=None):
        """Train model with generator

        Args:
            train_generator: generator of training data
            valid_generator: generator of validation data
            epochs: num of epochs to train the model
            callbacks: List of `keras.callbacks.Callback` instances to apply during training.

        """
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')

        logging.info('Training start...')
        self.model.fit_generator(generator=train_generator, epochs=epochs,
                                 validation_data=valid_generator, callbacks=callbacks)
        logging.info('Training end...')

    def predict(self, x):
        """Return model predictions

        Args:
            x: input for model

        Returns: np.array

        """
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')
        return self.model.predict(x)

    def save_model_json(self, json_file):
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')
        model_json = self.model.to_json()
        with open(json_file, 'w') as writer:
            writer.write(model_json)
        logging.info('Saved model architecture to disk:', json_file)

    def save_weights(self, weights_file):
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')
        self.model.save_weights(weights_file)
        logging.info('Saved model weights to disk:', weights_file)

    def save_model(self, json_file, weights_file):
        self.save_model_json(json_file)
        self.save_weights(weights_file)

    def load_model_json(self, json_file):
        with open(json_file, 'r') as reader:
            self.model = model_from_json(reader.read(), custom_objects=self.custom_objects)
        logging.info('Load model architecture from disk:', self.json_file)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)
        logging.info('Load model weights from disk:', weights_file)

    def load_model(self, json_file, weights_file):
        self.load_model_json(json_file)
        self.model.load_weights(weights_file)

    def load_best_model(self):
        """Load best model after using ModelCheckpoint callback"""
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')
        self.load_weights(self.weights_file)

    def load_swa_model(self):
        """Load swa model after using SWA callback"""
        if self.model is None:
            raise Exception('Model is None. Call build() method to build model first!')
        self.model.load_weights(self.swa_weights_file)
