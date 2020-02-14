# -*- coding: utf-8 -*-

import math
from typing import List, Optional

import tensorflow as tf
import numpy as np

from fancy_nlp.preprocessors import NERPreprocessor


class NERGenerator(tf.keras.utils.Sequence):
    """Data Generator for NER
    """
    def __init__(self,
                 preprocessor: NERPreprocessor,
                 data: List[List[str]],
                 labels: Optional[List[List[str]]] = None,
                 batch_size: int = 32,
                 shuffle: bool = True) -> None:
        """
        Args:
            preprocessor: Instance of NERPreprocessor, which helps to prepare feature input for
                ner model.
            data: List of List of str. List of tokenized texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            labels: List of List of str. The labels of train_data, usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
            batch_size: int. How many samples to train on in one iteration
            shuffle: Boolean. wWhether to shuffle data after each epoch of training.
        """
        self.preprocessor = preprocessor
        self.data = data
        self.labels = labels
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(math.ceil(self.data_size / self.batch_size))
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        if self.labels is not None:
            batch_data, batch_labels = zip(*[(self.data[i], self.labels[i]) for i in batch_index])
        else:
            batch_data = [self.data[i] for i in batch_index]
            batch_labels = None
        return self.preprocessor.prepare_input(batch_data, batch_labels)


class TextClassificationGenerator(tf.keras.utils.Sequence):
    """Data Generator for text classification
    """
    def __init__(self, preprocessor, data, labels=None, batch_size=32, shuffle=True):
        """
        Args:
            preprocessor: `TextClassificationPreprocessor` instance to help prepare input for ner model
            data: list of tokenized texts (, like ``[['我', '是', '中', '国', '人']]``
            labels: list of str, the corresponding label strings
            batch_size: how many samples to train on in one iteration
            shuffle: whether to shuffle data after each epoch of training
        """
        self.preprocessor = preprocessor
        self.data = data
        self.labels = labels
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(math.ceil(self.data_size / self.batch_size))
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        if self.labels is not None:
            batch_data, batch_labels = zip(*[(self.data[i], self.labels[i]) for i in batch_index])
        else:
            batch_data = [self.data[i] for i in batch_index]
            batch_labels = None
        return self.preprocessor.prepare_input(batch_data, batch_labels)


class SPMGenerator(tf.keras.utils.Sequence):
    """Data Generator for SPM
    """
    def __init__(self, preprocessor, data, labels=None, batch_size=32, shuffle=True):
        """
        Args:
            preprocessor: `SPMPreprocessor` instance to help prepare input for spm model
            data: list of text pairs (, like ``[['我是中国人', ...], ['我爱中国', ...]]``
            labels: list of str, the corresponding label strings
            batch_size: how many samples to train on in one iteration
            shuffle: whether to shuffle data after each epoch of training
        """
        self.preprocessor = preprocessor
        self.data = data
        self.labels = labels
        self.data_size = len(self.data[0])
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(math.ceil(self.data_size / self.batch_size))
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        if self.labels is not None:
            batch_data_a, batch_data_b, batch_labels = \
                zip(*[(self.data[0][i], self.data[1][i], self.labels[i]) for i in batch_index])
        else:
            batch_data_a, batch_data_b = zip(*[(self.data[0][i], self.data[1][i])
                                               for i in batch_index])
            batch_labels = None
        return self.preprocessor.prepare_input((batch_data_a, batch_data_b), batch_labels)
