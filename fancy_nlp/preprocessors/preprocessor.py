# -*- coding: utf-8 -*-

from absl import logging
from keras.preprocessing.sequence import pad_sequences

from ..utils import load_pre_trained, train_w2v, train_fasttext


class Preprocessor(object):
    """Base Preprocessor.
    """

    def __init__(self,
                 max_len=None,
                 padding_mode='post',
                 truncating_mode='post'):
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.cls_token = '<CLS>'
        self.seq_token = '<SEQ>'

        self.max_len = max_len
        self.padding_mode = padding_mode
        self.truncating_mode = truncating_mode

    @staticmethod
    def build_corpus(untokenized_texts, cut_func):
        """Build corpus from untokenized texts.

        Args:
            cut_func: function to tokenize texts. For example, lambda x: list(x) can be used for
                      tokenize texts in char level, while lambda x: jieba.lcut(x) can be used for
                      tokenize Chinese texts in word level.

        Returns: list of tokenized texts, like ``[['我', '是', '中', '国', '人']]``

        """
        corpus = []
        for text in untokenized_texts:
            corpus.append(cut_func(text))
        return corpus

    def build_vocab(self, corpus, min_count=3, special_token='standard'):
        """Build vocabulary using corpus.

        Args:
            corpus: list of tokenized texts, like ``[['我', '是', '中', '国', '人']]``
            min_count: token of which frequency is less than min_count will be ignored
            special_token: str, how to handle special tokens. If special_token is 'standard', we
                           add 2 special tokens: [('<PAD>', 0), ('<UNK>', 1)]. If special_token is
                           'bert', we add 4 special tokens: [('<PAD>', 0), ('<UNK>', 1),
                           ('<CLS>', 2), ('<SEQ>', 3)]

        Returns: tuple(dict, dict, dict):
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
        """Build label vocabulary
        """
        raise NotImplementedError

    def build_embedding(self, embed_type, vocab, corpus=None, embedding_dim=300,
                        special_token='standard'):
        """Prepare embeddings for the words in vocab.
        We support loading external pre-trained embeddings as well as training on the corpus to
        obtain embeddings

        Args:
            embed_type: str, can be a path to pre-trained embedding file or pre-train embedding
                        method to train on corpus
            vocab: a mapping of words to indices
            corpus: a list of tokenized texts
            embed_dim: dimensionality of embedding
            special_token: str, how to handle special tokens. If special_token is 'standard', we
                           add 2 special tokens: [('<PAD>', 0), ('<UNK>', 1)]. If special_token is
                           'bert', we add 4 special tokens: [('<PAD>', 0), ('<UNK>', 1),
                           ('<CLS>', 2), ('<SEQ>', 3)].
                           We will use zero-initializer for '<PAD>' token and random-initializer
                           for other special tokens.
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
                return load_pre_trained(embed_type, vocab, zero_init_indices, rand_init_indices)
            except FileNotFoundError:
                raise ValueError('Argument `embed_type` input error: {}'.format(embed_type))

    def prepare_input(self, data, label=None):
        """Prepare input for neural model training, evaluating and testing
        """
        raise NotImplementedError

    @staticmethod
    def build_id_sequence(tokenized_text, vocabulary, unk_idx=1):
        """Given a token list, return the corresponding id sequence.

        Args:
            tokenized_text: list of str, like `['我', '是', '中', '国', '人']`
            vocabulary: a mapping of words to indices
            unk_idx: the index of words that do not appear in vocabulary, we usually set it to 1

        Returns: list of indices

        """
        return [vocabulary.get(token, unk_idx) for token in tokenized_text]

    @staticmethod
    def build_id_matrix(tokenized_texts, vocabulary, unk_idx=1):
        """Given a list, each item is a token list, return the corresponding id matrix.

        Args:
            tokenized_texts: list of tokenized texts, like ``[['我', '是', '中', '国', '人']]``
            vocabulary: a mapping of words to indices
            unk_idx: the index of words that do not appear in vocabulary, we usually set it to 1

        Returns: list of list of indices

        """
        return [[vocabulary.get(token, unk_idx) for token in text] for text in tokenized_texts]

    def pad_sequence(self, sequence_list):
        """Given a list, each item is a id sequence, return the padded sequence
        """
        return pad_sequences(sequence_list, maxlen=self.max_len, padding=self.padding_mode,
                             truncating=self.truncating_mode)

    def label_decode(self, predictions):
        """Decode model predictions to labels
        """
        raise NotImplementedError
