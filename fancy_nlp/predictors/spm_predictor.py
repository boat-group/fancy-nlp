# -*- coding: utf-8 -*-

import numpy as np
from absl import logging


class SPMPredictor(object):
    """SPM predictor for evaluating spm model, output predictive probabilities
       for input sentence
    """
    def __init__(self, model, preprocessor):
        """

        Args:
            model: instance of keras model
            preprocessor: `SPMPreprocessor` instance to prepare feature input for spm model
        """
        self.model = model
        self.preprocessor = preprocessor

    def predict_prob(self, text):
        """Return probabilities for a pair of sentence

        Args:
            text: a pair of untokenized text(str)

        Returns: np.array, shaped [num_classes]

        """
        assert isinstance(text, list) and len(text) == 2, "input must be a list of two texts"
        features, _ = self.preprocessor.prepare_input([[text[0]], [text[1]]])
        pred_probs = self.model.predict(features)
        return pred_probs[0]

    def predict_prob_batch(self, texts):
        """Return probabilities for a batch sentence pairs

        Args:
            texts: a list of text pairs, each text must be untokenized

        Returns: np.array, shaped [num_texts, num_classes]
        """
        assert isinstance(texts, (list, tuple)) and len(texts) == 2, "input must be text pairs"
        features, _ = self.preprocessor.prepare_input(texts)
        pred_probs = self.model.predict(features)
        return pred_probs

    def matching(self, text):
        """Return label string for a pair of text

        Args:
            text: can be untokenized text pair

        Returns: str

        """
        pred_prob = self.predict_prob(text)
        labels = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0),
                                                self.preprocessor.label_dict)
        return labels[0]

    def matching_batch(self, texts):
        """Return label string for a batch text pairs

        Args:
            texts: a list of text pairs, each text must be untokenized

        Returns: list of str

        """
        pred_probs = self.predict_prob_batch(texts)
        labels = self.preprocessor.label_decode(pred_probs,
                                                self.preprocessor.label_dict)
        return labels

    def matching_with_prob(self, text):
        """Return the classification result for one sentence with probability

        Args:
            text: can be untokenized text pair

        Returns: tuple

        """
        pred_result = self.predict_prob(text)
        tags = self.preprocessor.label_decode(np.expand_dims(pred_result, 0),
                                              self.preprocessor.label_dict)
        label_name = tags[0]
        label_prob = np.max(pred_result)
        return label_name, pred_result

    def matching_with_prob_batch(self, text):
        """Return the matching results for a batch sentence pairs with probabilities

        Args:
            text: a list of text pairs, each text must be untokenized

        Returns: list of tuple

        """
        pred_results = self.predict_prob_batch(text)
        tags = self.preprocessor.label_decode(pred_results, self.preprocessor.label_dict)
        results = list(zip(tags, pred_results))
        return results
