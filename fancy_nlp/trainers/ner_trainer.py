# -*- coding: utf-8 -*-

import os
from typing import List, Optional

import tensorflow as tf
from absl import logging
from seqeval import metrics

from fancy_nlp.utils import NERGenerator
from fancy_nlp.callbacks import NERMetric, SWA
from fancy_nlp.preprocessors import NERPreprocessor


class NERTrainer(object):
    """NER Trainer, which is used to
    1) train ner model with given training dataset
    2) evaluate ner model with given validation dataset

    """

    def __init__(self,
                 model: tf.keras.models.Model,
                 preprocessor: NERPreprocessor) -> None:
        """

        Args:
            model: Instance of tf.keras Model. The ner model to be trained.
            preprocessor: Instance of NERPreprocessor, which helps to prepare feature input for
                ner model.
        """
        self.model = model
        self.preprocessor = preprocessor

    def train(self,
              train_data: List[List[str]],
              train_labels: List[List[str]],
              valid_data: Optional[List[List[str]]] = None,
              valid_labels: List[List[str]] = None,
              batch_size: int = 32,
              epochs: int = 50,
              callback_list: Optional[List[str]] = None,
              checkpoint_dir: Optional[str] = None,
              model_name: Optional[str] = None,
              swa_model: Optional[tf.keras.models.Model] = None,
              load_swa_model: bool = False) -> None:
        """Train ner model with provided training dataset. If validation dataset is provided,
        evaluate ner model with it after training.

        Args:
            train_data: List of List of str. List of tokenized (in char level) texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            train_labels: List of List of str. The labels of train_data, usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
            valid_data: Optional List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation.
            valid_labels: Optional List of List of str, can be None. The labels of valid_data.
                We can use fancy_nlp.utils.load_ner_data_and_labels() function to get training
                or validation data and labels from raw dataset in CoNLL format.
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model
            callback_list: Optional List of str or instance of `keras.callbacks.Callback`,
                can be None. Each item indicates the callback to apply during training. Currently,
                we support using 'modelcheckpoint' for `ModelCheckpoint` callback, 'earlystopping`
                for 'Earlystopping` callback, 'swa' for 'SWA' callback. We will automatically add
                `NERMetric` callback when valid_data and valid_labels are both provided.
            checkpoint_dir: Optional str, can be None. Directory to save the ner model. It must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training.
            model_name: Optional str, can be None. Prefix of ner model's weights file. I must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training. For example, if checkpoint_dir is 'ckpt' and
                model_name is 'model', the weights of ner model saved by `ModelCheckpoint` callback
                will be 'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'.
            swa_model: Instance of `tf.keras.model.Model`. The ner model which is used in `SWA`
                callback to keep track of weight averaging during training. It has the same architecture as
                self.model. Only pass it when using `SWA` callback.
            load_swa_model: Boolean. Whether to load swa model, only apply when using `SWA`
                Callback. We suggest set it to True when using `SWA` Callback since swa model
                performs better than the original model at most cases.

        """
        callbacks = self.prepare_callback(callback_list, valid_data, valid_labels, checkpoint_dir,
                                          model_name, swa_model)

        train_features, train_y = self.preprocessor.prepare_input(train_data, train_labels)
        if valid_data is not None and valid_labels is not None:
            valid_features, valid_y = self.preprocessor.prepare_input(valid_data, valid_labels)
            validation_data = (valid_features, valid_y)
        else:
            validation_data = None

        logging.info('Training start...')
        self.model.fit(x=train_features, y=train_y, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, callbacks=callbacks)
        logging.info('Training end...')

        if load_swa_model and callback_list is not None and 'swa' in callback_list:
            logging.info('Loading swa model after using SWA callback')
            self.load_model_weights(os.path.join(checkpoint_dir, f'{model_name}_swa.hdf5'))

        elif callback_list is not None and 'modelcheckpoint' in callback_list:
            logging.info('Loading best model after using ModelCheckpoint callback...')
            self.load_model_weights(os.path.join(checkpoint_dir, f'{model_name}.hdf5'))

    def train_generator(self,
                        train_data: List[List[str]],
                        train_labels: List[List[str]],
                        valid_data: Optional[List[List[str]]] = None,
                        valid_labels: Optional[List[List[str]]] = None,
                        batch_size: int = 32,
                        epochs: int = 50,
                        callback_list: Optional[List[str]] = None,
                        checkpoint_dir: Optional[str] = None,
                        model_name: Optional[str] = None,
                        swa_model: Optional[tf.keras.models.Model] = None,
                        load_swa_model=False):
        """Train the ner model with provided training data, using a generator that yields data
        batch-by-batch. If validation data is provided, evaluate the ner model with it
        after training.

        Args:
            train_data: List of List of str. List of tokenized (in char level) texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            train_labels: List of List of str. The labels of train_data, usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
            valid_data: Optional List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation.
            valid_labels: Optional List of List of str, can be None. The labels of valid_data.
                We can use fancy_nlp.utils.load_ner_data_and_labels() function to get training
                or validation data and labels from raw dataset in CoNLL format.
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model
            callback_list: Optional List of str or instance of `keras.callbacks.Callback`,
                can be None. Each item indicates the callback to apply during training. Currently,
                we support using 'modelcheckpoint' for `ModelCheckpoint` callback, 'earlystopping`
                for 'Earlystopping` callback, 'swa' for 'SWA' callback. We will automatically add
                `NERMetric` callback when valid_data and valid_labels are both provided.
            checkpoint_dir: Optional str, can be None. Directory to save the ner model. It must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training.
            model_name: Optional str, can be None. Prefix of ner model's weights file. I must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training. For example, if checkpoint_dir is 'ckpt' and
                model_name is 'model', the weights of ner model saved by `ModelCheckpoint` callback
                will be 'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'.
            swa_model: Instance of `tf.keras.model.Model`. The ner model which is used in `SWA`
                callback to keep track of weight averaging during training. It has the same architecture as
                self.model. Only pass it when using `SWA` callback.
            load_swa_model: Boolean. Whether to load swa model, only apply when using `SWA`
                Callback. We suggest set it to True when using `SWA` Callback since swa model
                performs better than the original model at most cases.

        """

        callbacks = self.prepare_callback(callback_list, valid_data, valid_labels, checkpoint_dir,
                                          model_name, swa_model)

        train_generator = NERGenerator(self.preprocessor, train_data, train_labels, batch_size)

        if valid_data and valid_labels:
            valid_generator = NERGenerator(self.preprocessor, valid_data, valid_labels,
                                           batch_size)
        else:
            valid_generator = None

        print('Training start...')
        # Note: Model.fit now supports generators
        self.model.fit(x=train_generator,
                       epochs=epochs,
                       callbacks=callbacks,
                       validation_data=valid_generator)
        print('Training end...')

        if load_swa_model and callback_list is not None and 'swa' in callback_list:
            logging.info('Loading swa model after using SWA callback')
            self.load_model_weights(os.path.join(checkpoint_dir, f'{model_name}_swa.hdf5'))

        elif callback_list is not None and 'modelcheckpoint' in callback_list:
            logging.info('Loading best model after using ModelCheckpoint callback...')
            self.load_model_weights(os.path.join(checkpoint_dir, f'{model_name}.hdf5'))

    def prepare_callback(self,
                         callback_list: List[str],
                         valid_data: Optional[List[List[str]]] = None,
                         valid_labels: Optional[List[List[str]]] = None,
                         checkpoint_dir: Optional[str] = None,
                         model_name: Optional[str] = None,
                         swa_model: Optional[tf.keras.models.Model] = None) \
            -> List[tf.keras.callbacks.Callback]:
        """Prepare the callbacks to be applied during training.

        Args:
            callback_list: List of str or instance of `keras.callbacks.Callback`. Each item
                indicates the callback to be applied during training. Currently, we support using
                'modelcheckpoint' for `ModelCheckpoint` callback, 'earlystopping` for
                'Earlystopping` callback, 'swa' for 'SWA' callback.
            valid_data: Optional List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation, like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            valid_labels: Optional List of List of str, can be None. The labels of valid_data,
                usually in BIO or BIOES format, like
                ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
                When valid_data and valid_labels are both provided, we will automatically add
                `NERMetric` callback for evaluation during training.
            checkpoint_dir: Optional str, can be None. Directory to save the ner model. It must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training.
            model_name: Optional str, can be None. Prefix of ner model's weights file. I must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training. For example, if checkpoint_dir is 'ckpt' and
                model_name is 'model', the weights of ner model saved by `ModelCheckpoint` callback
                will be 'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'.
            swa_model: Instance of `tf.keras.model.Model`. The ner model which is used in `SWA`
                callback to keep track of weight averaging during training. It has the same
                architecture as self.model. Only pass it when using `SWA` callback.

        Returns: List of `keras.callbacks.Callback` instances

        """
        assert not isinstance(callback_list, str)
        callback_list = callback_list or []
        callbacks = []
        if valid_data is not None and valid_labels is not None:
            callbacks.append(NERMetric(self.preprocessor, valid_data, valid_labels))
            add_metric = True
        else:
            add_metric = False

        if 'modelcheckpoint' in callback_list:
            if not add_metric:
                logging.warning(
                    'Using `ModelCheckpoint` without validation data provided is not Recommended! '
                    'We will use `loss` (of training data) as monitor.')

            assert checkpoint_dir is not None, \
                '`checkpoint_dir` must must be provided when using "ModelCheckpoint" callback'
            assert model_name is not None, \
                '`model_name` must must be provided when using "ModelCheckpoint" callback'
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f'{model_name}.hdf5'),
                monitor='val_f1' if add_metric else 'loss',
                save_best_only=True,
                save_weights_only=True,
                mode='max' if add_metric else 'min',
                verbose=1))
            logging.info('ModelCheckpoint Callback added')

        if 'earlystopping' in callback_list:
            if not add_metric:
                logging.warning('Using `Earlystopping` with validation data not provided is not '
                                'Recommended! We will use `loss` (of training data) as monitor.')
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_f1' if add_metric else 'loss',
                mode='max' if add_metric else 'min',
                patience=5,
                verbose=1))
            logging.info('Earlystopping Callback added')

        if 'swa' in callback_list:
            assert checkpoint_dir is not None, \
                '`checkpoint_dir` must must be provided when using "SWA" callback'
            assert model_name is not None, \
                '`model_name` must must be provided when using "SWA" callback'
            assert swa_model is not None, \
                '`swa_model` must must be provided when using "SWA" callback'
            callbacks.append(SWA(swa_model=swa_model, checkpoint_dir=checkpoint_dir,
                                 model_name=model_name, swa_start=5))
            logging.info('SWA Callback added')

        return callbacks

    def load_model_weights(self, weights_file: str) -> None:
        """Load model's weights from disk, which is usually called after training done.

        Args:
            weights_file: str. Path to load model weights

        Returns:

        """
        self.model.load_weights(weights_file)

    def evaluate(self, data: List[List[str]], labels: List[List[str]]) -> float:
        """Evaluate the performance of ner model with given data and labels, and return the f1
        score.

        Args:
            data: List of List of str. List of tokenized (in char level) texts ,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.
            labels: List of List of str. The corresponding labels , usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.

        Returns:
            Float. The F1 score.

        """
        features, y = self.preprocessor.prepare_input(data, labels)
        pred_probs = self.model.predict(features)
        if self.preprocessor.use_bert:
            pred_probs = pred_probs[:, 1:-1, :]     # remove <CLS> and <SEQ>

        lengths = [min(len(label), pred_prob.shape[0])
                   for label, pred_prob in zip(labels, pred_probs)]
        y_pred = self.preprocessor.label_decode(pred_probs, lengths)

        r = metrics.recall_score(labels, y_pred)
        p = metrics.precision_score(labels, y_pred)
        f1 = metrics.f1_score(labels, y_pred)

        logging.info('Recall: {}, Precision: {}, F1: {}'.format(r, p, f1))
        logging.info(metrics.classification_report(labels, y_pred))

        return f1
