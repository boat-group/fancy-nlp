# -*- coding: utf-8 -*-

from typing import List, Tuple

import tensorflow as tf
from seqeval import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

from fancy_nlp.preprocessors import NERPreprocessor, SPMPreprocessor


class NERMetric(tf.keras.callbacks.Callback):
    """Callback for evaluating ner model during training.
    """
    def __init__(self,
                 preprocessor: NERPreprocessor,
                 valid_data: List[List[str]],
                 valid_labels: List[List[str]]) -> None:
        """
        Args:
            preprocessor: Instance of `NERPreprocessor`, which helps to prepare feature input for
                ner model.
            valid_data: List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation, like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            valid_labels: List of List of str, can be None. The labels of valid_data, usually in
                BIO or BIOES format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
        """
        self.preprocessor = preprocessor
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.valid_features, self.valid_y = self.preprocessor.prepare_input(valid_data,
                                                                            valid_labels)
        super(NERMetric, self).__init__()

    def get_lengths(self, pred_probs):
        return [min(len(valid_label), pred_prob.shape[0])
                for valid_label, pred_prob in zip(self.valid_labels, pred_probs)]

    def on_epoch_end(self, epoch, logs=None):
        pred_probs = self.model.predict(self.valid_features)
        if self.preprocessor.use_bert:
            pred_probs = pred_probs[:, 1:-1, :]     # remove <CLS> and <SEQ>
        y_pred = self.preprocessor.label_decode(pred_probs, self.get_lengths(pred_probs))

        r = metrics.recall_score(self.valid_labels, y_pred)
        p = metrics.precision_score(self.valid_labels, y_pred)
        f1 = metrics.f1_score(self.valid_labels, y_pred)

        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch+1, r, p, f1))
        print(metrics.classification_report(self.valid_labels, y_pred))


class TextClassificationMetric(tf.keras.callbacks.Callback):
    """
    callback for evaluating text classification model
    """
    def __init__(self, preprocessor, valid_data, valid_labels):
        """
        Args:
            preprocessor: `TextClassificationPreprocessor` instance to help prepare input for
            text classification model
            valid_data: list of tokenized texts (, like ``[['我', '是', '中', '国', '人']]``
            valid_labels: list of str, the corresponding label strings
        """
        self.preprocessor = preprocessor
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.valid_features, self.valid_y = self.preprocessor.prepare_input(valid_data,
                                                                            valid_labels)
        super(TextClassificationMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred_probs = self.model.predict(self.valid_features)
        y_pred = self.preprocessor.label_decode(pred_probs)

        r = recall_score(self.valid_labels, y_pred, average='macro')
        p = precision_score(self.valid_labels, y_pred, average='macro')
        f1 = f1_score(self.valid_labels, y_pred, average='macro')

        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch + 1, r, p, f1))
        print(classification_report(self.valid_labels, y_pred))


class SPMMetric(tf.keras.callbacks.Callback):
    """
    callback for evaluating spm model
    """
    def __init__(self,
                 preprocessor: SPMPreprocessor,
                 valid_data: Tuple[List[str], List[str]],
                 valid_labels: List[str]) -> None:
        """
        Args:
            preprocessor: `SPMPreprocessor` instance to help prepare input for spm model
            valid_data: list of text pairs (, like ``[['我是中国人', ...], ['我爱中国', ...]]``
            valid_labels: list of str, the corresponding label strings
        """
        self.preprocessor = preprocessor
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.valid_features, self.valid_y = self.preprocessor.prepare_input(valid_data,
                                                                            valid_labels)
        super(SPMMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred_probs = self.model.predict(self.valid_features)
        y_pred = np.argmax(pred_probs, axis=-1)
        y_true = np.argmax(self.valid_y, axis=-1)

        r = recall_score(y_true, y_pred, average='macro')
        p = precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch, r, p, f1))
        print(classification_report(y_true, y_pred))
