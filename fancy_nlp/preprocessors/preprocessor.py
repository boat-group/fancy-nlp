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
        self.max_len = max_len
        self.padding_mode = padding_mode
        self.truncating_mode = truncating_mode

    @staticmethod
    def build_corpus(untokenized_texts, cut_func):
        """Build corpus.

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

    @staticmethod
    def build_vocab(corpus, min_count=3, start_index=2):
        """Build vocabulary using corpus.

        Args:
            corpus: list of tokenized texts, like ``[['我', '是', '中', '国', '人']]``
            min_count: token whose frequency is less than min_count will be ingnored
            start_index: token's starting index, we usually set it to be 2, which means we preserve
                         the first 2 indices: 0 for padding token, 1 for "unk" token

        Returns: tuple(dict, dict, dict):
                 1. token_count: a mapping of tokens to frequencies
                 2. token_vocab: a mapping of tokens to indices
                 3. id2token: a mapping of indices to tokens

        """
        token_count = {}
        for tokenized_text in corpus:
            for token in tokenized_text:
                token_count[token] = token_count.get(token, 0) + 1

        token_count = {token: count for token, count in token_count.items()
                       if count >= min_count}
        id2token = {idx + start_index: token for idx, token in enumerate(token_count)}
        token_vocab = {token: idx for idx, token in id2token.items()}

        logging.info('Build vocabulary finished, vocabulary size: {}'.format(len(token_vocab)))
        return token_count, token_vocab, id2token

    def build_label_vocab(self, labels):
        raise NotImplementedError

    @staticmethod
    def build_embedding(embed_type, vocab, corpus=None, pad_idx=0, unk_idx=1):
        """preprae embeddingd for the words in vocab
        """
        if embed_type is None:
            return None
        if embed_type == 'word2vec':
            return train_w2v(corpus, vocab, pad_idx, unk_idx)
        elif embed_type == 'fasttext':
            return train_fasttext(corpus, vocab, pad_idx, unk_idx)
        else:
            try:
                return load_pre_trained(embed_type, vocab, pad_idx, unk_idx)
            except FileNotFoundError:
                raise ValueError('`embed_type` input error: {}'.format(embed_type))

    @staticmethod
    def build_id_sequence(tokenized_texts, vocabulary, unk_idx=1):
        """Given a list, each item is a token list, return the corresponding id sequence.
        """
        return [[vocabulary.get(token, unk_idx) for token in text] for text in tokenized_texts]

    def pad_sequence(self, sequence_list):
        """Given a list, each item is a id sequence, return the padded sequence
        """
        return pad_sequences(sequence_list, maxlen=self.max_len, truncating=self.truncating_mode)
