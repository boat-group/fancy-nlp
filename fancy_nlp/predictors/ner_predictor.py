# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Union, List, Dict, Any

import numpy as np
from absl import logging
import tensorflow as tf
from seqeval.metrics import sequence_labeling

from fancy_nlp.preprocessors import NERPreprocessor


class NERPredictor(object):
    """NER predictor, which is used to
    1) output predictive probability sequence for given text;
    2) output predictive tag sequence for given text;
    3) output recognized entities with detailed information in pretty format for given text.

    """

    def __init__(self,
                 model: tf.keras.models.Model,
                 preprocessor: NERPreprocessor) -> None:
        """

        Args:
            model: instance of tf.keras model, the trained ner model.
            preprocessor: instance of `NERPreprocessor`, which helps to prepare feature input for
                ner model.

        """
        self.model = model
        self.preprocessor = preprocessor

    def predict_prob(self, text: Union[str, List[str]]) -> np.ndarray:
        """Return the probability sequence for given text predicted by the ner model

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``

        Returns: np.ndarray, shaped [num_chars, num_classes]

        """
        if isinstance(text, list):
            logging.warning('Text is passed in a list. Make sure it is tokenized in char level!')
            features, _ = self.preprocessor.prepare_input([text])
        else:
            assert isinstance(text, str)
            features, _ = self.preprocessor.prepare_input([list(text)])
        pred_probs = self.model.predict(features)

        if self.preprocessor.use_bert:
            # remove the probabilities of special tokens: <CLS> and <SEQ>
            return pred_probs[0, 1:-1, :]
        else:
            return pred_probs[0]

    def predict_prob_batch(self, texts: Union[List[str], List[List[str]]]) -> np.ndarray:
        """Return the probability sequences for given batch of text predicted by the ner model

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.

        Returns: np.ndarray, shaped [num_texts, num_chars, num_classes]

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
            # remove the probabilities of special tokens: <CLS> and <SEQ>
            return pred_probs[:, 1:-1, :]
        else:
            return pred_probs

    def tag(self, text: Union[str, List[str]]) -> List[str]:
        """Return the tag sequence of given text predicted by the ner model

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``

        Returns:
            List of str. The tag sequence, like ``['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']`
        """

        pred_prob = self.predict_prob(text)
        length = min(len(text), pred_prob.shape[0])
        tags = self.preprocessor.label_decode(np.expand_dims(pred_prob, 0), [length])
        return tags[0]

    def tag_batch(self, texts: Union[List[str], List[List[str]]]) -> List[List[str]]:
        """Return the tag sequences of given batch of texts predicted by the ner model

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.

        Returns:
            List of List of str. The tag sequences, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O',
            'O']]``

        """
        pred_probs = self.predict_prob_batch(texts)
        lengths = [min(len(text), pred_prob.shape[0]) for text, pred_prob in zip(texts, pred_probs)]
        tags = self.preprocessor.label_decode(pred_probs, lengths)
        return tags

    @staticmethod
    def entities(text: List[str], tag: List[str], pred_prob: np.ndarray) -> List[Dict[str, Any]]:
        """Return recognized entities with detailed information according to the tag sequence

        Args:
            text: List of str. A tokenized (in char level) text sequence,
                like ``['我', '在', '上', '海', '上', '学']``
            tag: List of str. The corresponding tag sequence of text,
                like ``['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']``
            pred_prob: np.ndarray, the probabilities of tag sequence, shaped [num_chars,]

        Returns:
            List of Dict. Each Dict contains the detailed information of each recognized entity (
            name, type, score, offset). Specifically, it will be like:
            [{'name': '上海',
              'type': 'LOC',
              'score': 0.9986118674278259,
              'beginOffset': 2,
              'endOffset': 4}
              ...
            ]

        """
        results = []
        chunks = sequence_labeling.get_entities(tag)

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
    def restrict_entities(text: List[str],
                          tag: List[str],
                          pred_prob: np.ndarray,
                          threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Return restricted entities according to tag sequence: 1) remove those entities of
        which scores are lower than threshold; 2) for each entity type, only keep the entity with
        the highest score.

        Args:
            text: List of str. A tokenized (in char level) text sequence,
                like ``['我', '在', '上', '海', '上', '学']``
            tag: List of str. The corresponding tag sequence of text,
                like ``['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']``
            pred_prob: np.ndarray, the probabilities of tag sequence, shaped [num_chars,]
            threshold: float. The scores of recognized entities must be higher than threshold.

        Returns:
            List of Dict. Each Dict contains the detailed information of each filtered entity (
            name, type, score, offset). Specifically, it will be like:
            [{'name': '上海',
              'type': 'LOC',
              'score': 0.9986118674278259,
              'beginOffset': 2,
              'endOffset': 4}
              ...
            ]

        """
        group_entities = defaultdict(list)

        chunks = sequence_labeling.get_entities(tag)
        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            score = float(np.average(pred_prob[chunk_start: chunk_end]))
            if score >= threshold:
                # remove entities of which scores are lower than threshold
                entity = ''.join(text[chunk_start: chunk_end])
                group_entities[chunk_type].append((entity, score, chunk_start, chunk_end))

        results = []
        for entity_type, group in group_entities.items():
            entity = sorted(group, key=lambda x: x[1])[-1]  # sorted by score
            results.append({
                'name': entity[0],
                'type': entity_type,
                'score': entity[1],
                'beginOffset': entity[2],
                'endOffset': entity[3]
            })
        return results

    def pretty_tag(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Analyze the tagging result of given text predicted by the ner model and return the
        result in pretty format with detailed information.

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``.

        Returns:
            A Dict including the original text and list of recognized entities with detailed
            information (name, type, score, offset). Specifically, it will be like:
            {'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
            }
            Note: the score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution.

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

    def pretty_tag_batch(self, texts: Union[List[str], List[List[str]]]) -> List[Dict[str, Any]]:
        """Analyze the tagging results of given batch of text predicted by the ner model and
        return the results in pretty format with detailed information.

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.

        Returns:
            List of Dict. Each Dict contain the tagging results of one text, including the original
            text and list of recognized entities with detailed information (name, type, score,
            offset). Specifically, it will be like:
            [{'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
             }
             ...
            ]
            Note: the score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution.

        """
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

    def restrict_tag(self,
                     text: Union[str, List[str]],
                     threshold: float = 0.85) -> Dict[str, Any]:
        """Analyze the tagging result of given text predicted by the ner model and then remove some
        recognized entities such that 1) all entities's scores are higher than threshold; 2)
        each entity type only keep one entity with the highest score. After that, return the
        recognized result in pretty format with detailed information.

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``.
            threshold: float. The scores of recognized entities must be higher than threshold.

        Returns:
            A Dict including the original text and list of recognized entities with detailed
            information (name, type, score, offset). Specifically, it will be like:
            {'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
            }

        Notes:
            The score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution. As a result,
            we do not recommend you use this function when using crf layer.

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

    def restrict_tag_batch(self,
                           texts: Union[List[str], List[List[str]]],
                           threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Analyze the tagging results of given batch of texts predicted by the ner model and then
        remove some recognized entities such that 1) all entities's scores are higher than
        threshold; 2) for each entity type, only keep the entity with the highest score. After
        that, return the recognized results in pretty format with detailed information.

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``
            threshold: float. The scores of recognized entities must be higher than threshold.

        Returns:
            List of Dict. Each Dict contain the tagging results of one text, including the original
            text and list of recognized entities with detailed information (name, type, score,
            offset). Specifically, it will be like:
            [{'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
             }
             ...
            ]

        Notes:
            The score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution. As a result,
            we do not recommend you use this function when using crf layer.

        """
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
