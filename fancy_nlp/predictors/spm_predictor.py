# -*- coding: utf-8 -*-

from typing import Tuple, List

import numpy as np
import tensorflow as tf

from fancy_nlp.preprocessors import SPMPreprocessor


class SPMPredictor(object):
    """SPM predictor for evaluating spm model, output predictive probabilities
       for input sentence
    """
    def __init__(self,
                 model: tf.keras.models.Model,
                 preprocessor: SPMPreprocessor) -> None:
        """

        Args:
            model: instance of keras model
            preprocessor: `SPMPreprocessor` instance to prepare feature input for spm model
        """
        self.model = model
        self.preprocessor = preprocessor

    def predict_prob(self, text: Tuple[str, str]) -> np.ndarray:
        """Return probabilities for a pair of sentence

        Args:
            text: a pair of untokenized text(str)

        Returns: np.array, shaped [num_classes]

        """
        assert isinstance(text, tuple) and len(text) == 2, "input must be a tuple of two texts"
        features, _ = self.preprocessor.prepare_input(([text[0]], [text[1]]))
        pred_probs = self.model.predict(features)
        return pred_probs[0]

    def predict_prob_batch(self, texts: Tuple[List[str], List[str]]) -> np.ndarray:
        """Return probabilities for a batch sentence pairs

        Args:
            texts: a list of text pairs, each text must be untokenized

        Returns: np.array, shaped [num_texts, num_classes]
        """
        assert isinstance(texts, (list, tuple)) and len(texts) == 2, "input must be text pairs"
        features, _ = self.preprocessor.prepare_input(texts)
        pred_probs = self.model.predict(features)
        return pred_probs

    def matching(self, text: Tuple[str, str]) -> str:
        """Return label string for a pair of text

        Args:
            text: can be untokenized text pair

        Returns: str

        """
        pred_prob = self.predict_prob(text)
        labels = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0))
        return labels[0]

    def matching_batch(self, texts: Tuple[List[str], List[str]]) -> List[str]:
        """Return label string for a batch text pairs

        Args:
            texts: a list of text pairs, each text must be untokenized

        Returns: list of str

        """
        pred_probs = self.predict_prob_batch(texts)
        labels = self.preprocessor.label_decode(pred_probs)
        return labels

    def matching_with_prob(self, text: Tuple[str, str]) -> Tuple[str, np.ndarray]:
        """Return the classification result for one sentence with probability

        Args:
            text: can be untokenized text pair

        Returns: tuple

        """
        pred_result = self.predict_prob(text)
        tags = self.preprocessor.label_decode(np.expand_dims(pred_result, 0))
        label_name = tags[0]
        return label_name, pred_result

    def matching_with_prob_batch(self, text: Tuple[List[str], List[str]]) -> \
        List[Tuple[str, np.ndarray]]:
        """Return the matching results for a batch sentence pairs with probabilities

        Args:
            text: a list of text pairs, each text must be untokenized

        Returns: list of tuple

        """
        pred_results = self.predict_prob_batch(text)
        tags = self.preprocessor.label_decode(pred_results)
        results = list(zip(tags, pred_results))
        return results
