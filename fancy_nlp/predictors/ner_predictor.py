# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from absl import logging
from seqeval.metrics.sequence_labeling import get_entities


class NERPredictor(object):
    """NER predictor for evaluating ner model, output predictive probabilities and predictive tag
    sequences for input sentence"""
    def __init__(self, model, preprocessor):
        """

        Args:
            model: instance of keras model
            preprocessor: `NERPreprocessor` instance to prepare feature input for ner model
        """
        self.model = model
        self.preprocessor = preprocessor

    def predict_prob(self, text):
        """Return probabilities for one sentence

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns: np.array, shaped [num_chars, num_classes]

        """
        if isinstance(text, list):
            logging.warning('Text is passed in a list. Make sure it is tokenized in char level!')
            features, _ = self.preprocessor.prepare_input([text])
        else:
            assert isinstance(text, str)
            features, _ = self.preprocessor.prepare_input([list(text)])
        pred_probs = self.model.predict(features)

        if self.preprocessor.use_bert:
            return pred_probs[0, 1:-1, :]
        else:
            return pred_probs[0]

    def predict_prob_batch(self, texts):
        """Return probabilities for a batch sentences

        Args:
            texts: a list of texts, each text can be untokenized (str) or
                   tokenized in char level (list)

        Returns: np.array, shaped [num_texts, num_chars, num_classes
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

        if self.preprocessor.use_bert:
            return pred_probs[:, 1:-1, :]
        else:
            return pred_probs

    def tag(self, text):
        """Return tag sequence for one sentence

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns: list of str

        """
        pred_prob = self.predict_prob(text)
        length = min(len(text), pred_prob.shape[0])
        tags = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0), [length])
        return tags[0]

    def tag_batch(self, texts):
        """Return tag sequences for a batch sentences

        Args:
            texts: a list of text, each text can be untokenized (str) or
                   tokenized in char level (list)

        Returns: list of list of str

        """
        pred_probs = self.predict_prob_batch(texts)
        lengths = [min(len(text), pred_prob.shape[0]) for text, pred_prob in zip(texts, pred_probs)]
        tags = self.preprocessor.label_decode(pred_probs, lengths)
        return tags

    @staticmethod
    def entities(text, tag, pred_prob):
        """Return entities according to tag sequence
        """
        results = []
        chunks = get_entities(tag)

        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            entity = {
                'name': ''.join(text[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(pred_prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            results.append(entity)
        return results

    @staticmethod
    def restrict_entities(text, tag, pred_prob, threshold=0.85):
        """Return restricted entities according to tag sequence: only keep at most one entity for
        each entity type
        """
        group_entities = defaultdict(list)

        chunks = get_entities(tag)
        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            score = float(np.average(pred_prob[chunk_start: chunk_end]))
            if score >= threshold:
                entity = ''.join(text[chunk_start: chunk_end])
                group_entities[chunk_type].append((entity, score, chunk_start, chunk_end))

        results = []
        for entity_type, group in group_entities.items():
            entity = sorted(group, key=lambda x: x[0])[-1]
            results.append({
                'name': entity[0],
                'type': entity_type,
                'score': entity[1],
                'beginOffset': entity[2],
                'endOffset': entity[3]
            })
        return results

    def pretty_tag(self, text):
        """Return tag sequence for one sentence in a pretty format

        Args:
            text: can be untokenized (str) or tokenized in char level (list)

        Returns:

        """
        pred_prob = self.predict_prob(text)
        length = min(len(text), pred_prob.shape[0])
        tag = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0), [length])

        pred_prob = np.max(pred_prob, axis=-1)
        char_cut = text if isinstance(text, list) else list(text)
        results = {
            'text': ''.join(char_cut),
            'entities': self.entities(char_cut, tag, pred_prob)
        }
        return results

    def pretty_tag_batch(self, texts):
        pred_probs = self.predict_prob_batch(texts)
        lengths = [min(len(text), pred_prob.shape[0]) for text, pred_prob in zip(texts, pred_probs)]
        tags = self.preprocessor.label_decode(pred_probs, lengths)

        pred_probs = np.max(pred_probs, axis=-1)
        results = []
        for text, tag, pred_prob in zip(texts, tags, pred_probs):
            char_cut = text if isinstance(text, list) else list(text)
            results.append({
                'text': ''.join(char_cut),
                'entities': self.entities(char_cut, tag, pred_prob)
            })
        return results

    def restrict_tag(self, text, threshold=0.85):
        """Return a restricted tag sequence for one sentence: only keep at most one entity
        for each entity type.

        Args:
            text: can be untokenized (str) or tokenized in char level (list)
            threshold:
        Returns:

        """
        pred_prob = self.predict_prob(text)
        length = min(len(text), pred_prob.shape[0])
        tag = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0), [length])

        pred_prob = np.max(pred_prob, axis=-1)
        char_cut = text if isinstance(text, list) else list(text)
        results = {
            'text': ''.join(char_cut),
            'entities': self.restrict_entities(char_cut, tag, pred_prob, threshold)
        }
        return results

    def restrict_tag_batch(self, texts, threshold=0.85):
        pred_probs = self.predict_prob_batch(texts)
        lengths = [min(len(text), pred_prob.shape[0]) for text, pred_prob in zip(texts, pred_probs)]
        tags = self.preprocessor.label_decode(pred_probs, lengths)

        pred_probs = np.max(pred_probs, axis=-1)
        results = []
        for text, tag, pred_prob in zip(texts, tags, pred_probs):
            char_cut = text if isinstance(text, list) else list(text)
            results.append({
                'text': ''.join(char_cut),
                'entities': self.restrict_entities(char_cut, tag, pred_prob, threshold)
            })
        return results
