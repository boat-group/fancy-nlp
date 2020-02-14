# -*- coding: utf-8 -*-

from typing import Callable, List, Tuple, Dict, Optional

from absl import logging
import tensorflow as tf
import numpy as np

from ..utils import load_pre_trained, train_w2v, train_fasttext


class Preprocessor(object):
    """Basic class for Fancy-NLP Preprocessor. All the preprocessor will inherit from it.

    Preprocessor is used to
    1) build vocabulary from training data;
    2) pre-trained embedding matrix using training corpus;
    3) prepare feature input for model;
    4) decode model predictions to label string

    """

    def __init__(self,
                 max_len: int = None,
                 padding_mode: str = 'post',
                 truncating_mode: str = 'post') -> None:
        """

        Args:
            max_len: Optional int, can be None. Max length of one sequence.
            padding_mode: str. 'pre' or 'post': pad either before or after each sequence, used when
                preparing feature input for model.
            truncating_mode: str. pre' or 'post': remove values from sequences larger than
                `maxlen`, either at the beginning or at the end of the sequences, used when
                preparing feature input for model.
        """
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.cls_token = '<CLS>'
        self.seq_token = '<SEQ>'

        self.max_len = max_len
        self.padding_mode = padding_mode
        self.truncating_mode = truncating_mode

    @staticmethod
    def build_corpus(untokenized_texts: List[str],
                     cut_func: Callable[[str], List[str]]) -> List[List[str]]:
        """Build corpus from untokenized texts.

        Args:
            untokenized_texts: List of str. List of un-tokenized texts, like ['我是中国人', ...].
            cut_func: Function to tokenize texts. For example, cut_func=lambda x: list(x) can be
                used for tokenize texts in char level, while cut_func=lambda x: jieba.lcut(x) can
                be used for tokenize Chinese texts in word level.

        Returns:
            List of List of str. List of tokenized texts, like
            ``[['我', '是', '中', '国', '人'], ...]`.`

        """
        corpus = []
        for text in untokenized_texts:
            corpus.append(cut_func(text))
        return corpus

    def build_vocab(self,
                    corpus: List[List[str]],
                    min_count: int = 3,
                    special_token: str = 'standard') \
            -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str]]:
        """Build vocabulary using corpus.

        Args:
            corpus: List of List of str. List of tokenized texts, like
                ``[['我', '是', '中', '国', '人'], ...]``
            min_count: int. Token of which frequency is less than min_count will be ignored
            special_token: str. 'standard' or 'bert': determine how to handle special tokens.
                If special_token is 'standard', we add 2 special tokens: [('<PAD>', 0), ('<UNK>',
                1)]. If special_token is 'bert', we add 4 special tokens: [('<PAD>', 0),
                ('<UNK>', 1), ('<CLS>', 2), ('<SEQ>', 3)]

        Returns: Tuple of 3 dicts :
                 1. token_count: a mapping of tokens to frequencies
                 2. token_vocab: a mapping of tokens to indices
                 3. id2token: a mapping of indices to tokens

        """
        if special_token == 'standard':
            token_vocab = {self.pad_token: 0,
                           self.unk_token: 1}
        elif special_token == 'bert':
            token_vocab = {self.pad_token: 0,
                           self.unk_token: 1,
                           self.cls_token: 2,
                           self.seq_token: 3}
        else:
            raise ValueError('Argument `special_token` can only be "standard" or "bert", '
                             'got: {}'.format(special_token))

        token_count = {}
        for tokenized_text in corpus:
            for token in tokenized_text:
                token_count[token] = token_count.get(token, 0) + 1
        # filter out low-frequency token
        token_count = {token: count for token, count in token_count.items()
                       if count >= min_count}

        for token in token_count:
            token_vocab[token] = len(token_vocab)
        id2token = dict((idx, token) for token, idx in token_vocab.items())

        logging.info('Build vocabulary finished, vocabulary size: {}'.format(len(token_vocab)))
        return token_count, token_vocab, id2token

    def build_label_vocab(self, labels):
        """Build label vocabulary.
        """
        raise NotImplementedError

    def build_embedding(self,
                        embed_type: Optional[str],
                        vocab: Dict[str, int],
                        corpus: Optional[List[List[str]]] = None,
                        embedding_dim: int = 300,
                        special_token: str = 'standard'):
        """Prepare embeddings for the tokens in vocab.
        We support loading external pre-trained embeddings with procided embedding file as well as
        training embeddings on the provied corpus.

        Args:
            embed_type: Optional str, can be None. The type of embedding, can be a
                pre-trained embedding filename that used to load pre-trained embedding,
                or a embedding training method (one of {'word2vec', 'fasttext'}) that used to
                train character embedding with dataset. If None, do not apply anr pre-trained
                embedding, and use randomly initialized embedding instead.
            vocab: Dict[str, int]. A mapping of tokens to indices
            corpus: List of tokenized texts,, like ``[['我', '是', '中', '国', '人'], ...]`.
            embedding_dim: int. Dimensionality of embedding
            special_token: str. 'standard' or 'bert': determine how to handle special tokens.
                If special_token is 'standard', we add 2 special tokens: [('<PAD>', 0), ('<UNK>',
                1)]. If special_token is 'bert', we add 4 special tokens: [('<PAD>', 0),
                ('<UNK>', 1), ('<CLS>', 2), ('<SEQ>', 3)]
                We will use zero-initializer for '<PAD>' token and random-initializer for other
                special tokens.
        """
        zero_init_indices = vocab.get(self.pad_token)
        if special_token == 'standard':
            rand_init_indices = vocab.get(self.unk_token)
        elif special_token == 'bert':
            rand_init_indices = [vocab.get(self.unk_token),
                                 vocab.get(self.cls_token),
                                 vocab.get(self.seq_token)]
        else:
            raise ValueError('Argument `special_token` can only be "standard" or "bert", '
                             'got: {}'.format(special_token))

        if embed_type is None:
            return None     # do not adopt any pre-trained embeddings
        if embed_type == 'word2vec':
            return train_w2v(corpus, vocab, zero_init_indices, rand_init_indices, embedding_dim)
        elif embed_type == 'fasttext':
            return train_fasttext(corpus, vocab, zero_init_indices, rand_init_indices,
                                  embedding_dim)
        else:
            try:
                return load_pre_trained(embed_type, embedding_dim, vocab,
                                        zero_init_indices, rand_init_indices)
            except FileNotFoundError:
                raise ValueError('Argument `embed_type` input error: {}'.format(embed_type))

    def prepare_input(self, data, label=None):
        """Prepare feature input for neural model training, evaluating and testing.
        """
        raise NotImplementedError

    @staticmethod
    def build_id_sequence(tokenized_text: List[str],
                          vocabulary: Dict[str, int],
                          unk_idx: int = 1) -> List[int]:
        """Given a token list, return the corresponding id sequence.

        Args:
            tokenized_text: List of str, like `['我', '是', '中', '国', '人']`.
            vocabulary: Dict[str, int]. A mapping of tokens to indices.
            unk_idx: int. The index of tokens that do not appear in vocabulary. We usually set it
                to 1.

        Returns:
            List of indices.

        """
        return [vocabulary.get(token, unk_idx) for token in tokenized_text]

    @staticmethod
    def build_id_matrix(tokenized_texts: List[List[str]],
                        vocabulary, unk_idx=1):
        """Given a list, each item is a token list, return the corresponding id matrix.

        Args:
            tokenized_texts: List of List of str. List of tokenized texts, like ``[['我', '是', '中',
                '国', '人'], ...]``.
            vocabulary: Dict[str, int]. A mapping of tokens to indices
            unk_idx: int. The index of tokens that do not appear in vocabulary. We usually set it
                to 1.

        Returns:
            List of List of indices

        """
        return [[vocabulary.get(token, unk_idx) for token in text] for text in tokenized_texts]

    def pad_sequence(self,
                     sequence_list: List[List[int]]) -> np.ndarray:
        """Given a list, each item is a id sequence, return the padded sequence.

        Args:
            sequence_list: List of List of int, where each element is a sequence.

        Returns:
            a 2D Numpy array of shape `(num_samples, num_timesteps)`

        """
        return tf.keras.preprocessing.sequence.pad_sequences(sequence_list,
                                                             maxlen=self.max_len,
                                                             padding=self.padding_mode,
                                                             truncating=self.truncating_mode)

    def label_decode(self, predictions, label_dict):
        """Decode model predictions to label strings
        """
        raise NotImplementedError
