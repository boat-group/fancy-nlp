# -*- coding: utf-8 -*-

import os
from absl import logging
from keras.callbacks import *
from fancy_nlp.utils import NERGenerator
from fancy_nlp.callbacks import NERMetric
from fancy_nlp.callbacks import SWA


class NERTrainer(object):
    def __init__(self, model, preprocessor):
        """

        Args:
            model: instance of NER Model
            preprocessor: instance of NERPreporcessor
        """
        self.model = model
        self.preprocessor = preprocessor

    def prepare_callback(self, callbacks_str, valid_data=None, valid_labels=None):
        """

        Args:
            callbacks_str: list of str, each item indicate the callback to apply during training.
                       For example, 'earlystopping' means using 'EarlyStopping' callback.
            valid_data:
            valid_labels:

        Returns: a list of `keras.callbacks.Callback` instances and a boolean variable indicate
                 whether modelchekpoint callback is added

        """

        callbacks_str = callbacks_str or []
        callbacks = []
        if valid_data is not None and valid_labels is not None:
            callbacks.append(NERMetric(self.preprocessor, valid_data, valid_labels))
            add_metric = True
        else:
            add_metric = False

        add_modelcheckpoint = False

        # ModelCheckpoint: must add
        if 'modelcheckpoint' in callbacks_str:
            if not add_metric:
                logging.warning('Using `ModelCheckpoint` with validation data not provided is not '
                                'Recommended! We will use `loss` (of training data) as monitor.')
            callbacks.append(ModelCheckpoint(filepath=self.model.weights_file,
                                             monitor='val_f1' if add_metric else 'loss',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode='max' if add_metric else 'min',
                                             verbose=1))
            logging.info('ModelCheckpoint Callback added')
            add_modelcheckpoint = True

        if 'earlystopping' in callbacks_str:
            if not add_metric:
                logging.warning('Using `Earlystopping` with validation data not provided is not '
                                'Recommended! We will use `loss` (of training data) as monitor.')
            callbacks.append(EarlyStopping(monitor='val_f1' if add_metric else 'loss',
                                           mode='max' if add_metric else 'min',
                                           patience=5,
                                           verbose=1))
            logging.info('Earlystopping Callback added')

        if 'swa' in callbacks_str:
            callbacks.append(SWA(swa_model=self.model.build_model(),
                                 checkpoint_dir=self.model.checkpoint_dir,
                                 model_name=self.model.model_name,
                                 swa_start=5))
            logging.info('SWA Callback added')

        return callbacks, add_modelcheckpoint

    def train(self, train_data, train_labels, valid_data=None, valid_labels=None,
              batch_size=32, epochs=50, callbacks_str=None):
        callbacks, add_modelcheckpoint = self.prepare_callback(callbacks_str, valid_data,
                                                               valid_labels)

        train_features, train_y = self.preprocessor.prepare_input(train_data, train_labels)
        if valid_data is not None and valid_labels is not None:
            valid_features, valid_y = self.preprocessor.prepare_input(valid_data, valid_labels)
        else:
            valid_features, valid_y = None, None

        self.model.fit(train_features, train_y, valid_features, valid_y, batch_size,
                       epochs, callbacks)
        if add_modelcheckpoint:
            self.model.load_best_model()

    def train_generator(self, train_data, train_labels, valid_data=None, valid_labels=None,
                        batch_size=32, epochs=50, callbacks_str=None, shuffle=True):
        print(valid_data is None)
        print(valid_labels is None)
        callbacks, add_modelcheckpoint = self.prepare_callback(callbacks_str, valid_data,
                                                               valid_labels)

        train_generator = NERGenerator(self.preprocessor, train_data, train_labels, batch_size,
                                       shuffle)

        if valid_data and valid_labels:
            valid_generator = NERGenerator(self.preprocessor, valid_data, valid_labels,
                                           batch_size, shuffle)
        else:
            valid_generator = None

        self.model.fit_generator(train_generator, valid_generator, epochs, callbacks)
        if add_modelcheckpoint:
            self.model.load_best_model()