# -*- coding: utf-8 -*-

import numpy as np
from absl import logging


class TextClassificationPredictor(object):
    """TextClassification predictor for evaluating text classification model, output predictive
    probabilities and labels for input sentence"""
    def __init__(self, model, preprocessor):
        """

        Args:
            model: instance of keras model
            preprocessor: `TextClassificationPreprocessor` instance
        """
        self.model = model
        self.preprocessor = preprocessor

    def predict_prob(self, text):
        """Return probabilities for one sentence

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns: np.array, shaped [num_classes,]

        """
        if isinstance(text, list):
            logging.warning('Text is passed in a list. Make sure it is tokenized in char level!')
            features, _ = self.preprocessor.prepare_input([text])
        else:
            assert isinstance(text, str)
            features, _ = self.preprocessor.prepare_input([list(text)])
        pred_probs = self.model.predict(features)
        return pred_probs[0]

    def predict_prob_batch(self, texts):
        """Return probabilities for a batch sentences

        Args:
            texts: a list of texts, each text can be untokenized (str) or
                   tokenized in char level (list)

        Returns: np.array, shaped [num_texts, num_classes]
        """
        assert isinstance(texts, list)
        if isinstance(texts[0], list):
            logging.warning('Text is passed in a list. Make sure it is tokenized in char level!')
            features, _ = self.preprocessor.prepare_input(texts)
        else:
            assert isinstance(texts[0], str)
            char_cut_texts = [list(text) for text in texts]
            features, _ = self.preprocessor.prepare_input(char_cut_texts)
        pred_probs = self.model.predict(features)
        return pred_probs

    def classify(self, text):
        """Return the classification result for one sentence

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns: str

        """
        pred_prob = self.predict_prob(text)
        tags = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0),
                                              self.preprocessor.label_dict)
        return tags[0]

    def classify_batch(self, texts):
        """Return classification result for a batch sentences

        Args:
            texts: a list of text, each text can be untokenized (str) or
                   tokenized in char level (list)

        Returns: list of str

        """
        pred_probs = self.predict_prob_batch(texts)
        tags = self.preprocessor.label_decode(pred_probs, self.preprocessor.label_dict)
        return tags

    def classification_with_prob(self, text):
        """Return the classification result for one sentence with probability

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns: tuple

        """
        pred_result = self.predict_prob(text)
        tags = self.preprocessor.label_decode(np.expand_dims(pred_result, 0),
                                              self.preprocessor.label_dict)
        label_name = tags[0]
        label_prob = np.max(pred_result)
        return label_name, label_prob

    def classification_with_prob_batch(self, text):
        """Return the classification results for a batch sentences with probabilities

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns: list of tuple

        """
        pred_results = self.predict_prob_batch(text)
        tags = self.preprocessor.label_decode(pred_results, self.preprocessor.label_dict)
        results = list(zip(tags, pred_results))
        return results
