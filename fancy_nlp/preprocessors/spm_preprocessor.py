# -*- coding: utf-8 -*-

import pickle
import codecs

import numpy as np
import jieba
from itertools import chain
from absl import logging
from keras.utils.np_utils import to_categorical

from fancy_nlp.preprocessors.preprocessor import Preprocessor
from fancy_nlp.utils import pad_sequences_2d, get_len_from_corpus, ChineseBertTokenizer


class SPMPreprocessor(Preprocessor):
    """SPM preprocessor.
    """
    def __init__(self, train_data, train_labels, min_count=2, use_word=False, use_char=True,
                 use_bert=False, use_bert_model=False, external_word_dict=None, bert_vocab_file=None,
                 word_embed_type=None, word_embed_dim=300, char_embed_type=None, char_embed_dim=300,
                 label_dict_file=None, max_len=None, max_word_len=None,
                 padding_mode='post', truncating_mode='post'):
        """

        Args:
            train_data: a list of untokenized text pairs
            train_labels: list of str, train_data's labels
            min_count: int, token of which frequency is lower than min_count will be ignored
            use_word: whether to use word embedding as input
            use_char：whether to use char embedding as input
            use_bert: whether to use bert embedding as input
            use_bert_model: boolean, whether to use traditional bert model which combines two
                            sentences as one input
            word_embed_type: str, can be a pre-trained embedding filename or pre-trained embedding
                             methods (word2vec, glove, fastext)
            word_embed_dim: dimensionality of word embedding
            char_embed_type: same as word_embed_type, only apply when use_char is True
            char_embed_dim: dimensionality of char embedding
            external_word_dict: external word dictionary, only apply when use_word is True
            bert_vocab_file: vocabulary file of pre-trained bert model, only apply when use_bert is
                             True
            label_dict_file: a file with two columns separated by tab, the first column is raw
                             label name, and the second column is the corresponding name which is
                             meaningful
            max_len: int, max sequence length
            max_word_len: int, max word length
            padding_mode: str, 'pre' or 'post', pad either before or after each sequence
            truncating_mode: str, 'pre' or 'post', remove values from sequences larger than
                             `max_len`, either at the beginning or at the end of the sequences
        """
        super(SPMPreprocessor, self).__init__(max_len, padding_mode, truncating_mode)

        self.train_data = train_data
        self.train_labels = train_labels
        self.min_count = min_count
        self.use_word = use_word
        self.use_char = use_char
        self.use_bert = use_bert
        self.use_bert_model = use_bert_model
        self.external_word_dict = external_word_dict
        self.word_embed_type = word_embed_type
        self.char_embed_type = char_embed_type
        self.max_word_len = max_word_len

        self.label_dict = self.load_label_dict(label_dict_file)

        assert not (self.use_bert_model and (self.use_word or self.use_char)), \
            "bert model can not add word or char embedding as additional input"
        assert not (self.use_bert_model and not use_bert), "bert model must use bert embedding"
        assert self.use_word or self.use_char or self.use_bert, "must use word or char or bert" \
                                                                "embedding as main input"
        assert not (self.use_word and self.use_bert), "bert embedding can not be used with word" \
                                                      "embedding"
        special_token = 'bert' if self.use_bert else 'standard'

        train_data_a, train_data_b = self.train_data
        train_data = list(chain(*zip(train_data_a, train_data_b)))

        # build word vocabulary and word embedding
        if self.use_word:
            self.load_word_dict()

            word_corpus = self.build_corpus(train_data, cut_func=lambda x: jieba.lcut(x))
            self.word_vocab_count, self.word_vocab, self.id2word = \
                self.build_vocab(word_corpus, self.min_count, special_token)
            self.word_vocab_size = len(self.word_vocab)
            self.word_embeddings = self.build_embedding(word_embed_type, self.word_vocab,
                                                        word_corpus, word_embed_dim,
                                                        special_token)
            if self.word_embeddings is not None:
                self.word_embed_dim = self.word_embeddings.shape[1]
            else:
                self.word_embed_dim = word_embed_dim

            if self.max_len is None:
                self.max_len = get_len_from_corpus(word_corpus)
            if self.use_char and self.max_word_len is None:
                self.max_word_len = get_len_from_corpus(list(chain(*word_corpus)))
        else:
            self.word_vocab_count, self.word_vocab, self.id2word = None, None, None
            self.word_vocab_size = -1
            self.word_embeddings = None
            self.word_embed_dim = -1

        train_data = [list(text) for text in train_data]

        # build char vocabulary and char embedding
        if self.use_char:
            self.char_vocab_count, self.char_vocab, self.id2char = \
                self.build_vocab(train_data, self.min_count, special_token)
            self.char_vocab_size = len(self.char_vocab)
            self.char_embeddings = self.build_embedding(char_embed_type, self.char_vocab,
                                                        train_data, char_embed_dim,
                                                        special_token)
            if self.char_embeddings is not None:
                self.char_embed_dim = self.char_embeddings.shape[1]
            else:
                self.char_embed_dim = char_embed_dim
            if self.max_len is None:
                self.max_len = get_len_from_corpus(train_data)
        else:
            self.char_vocab_count, self.char_vocab, self.id2char = None, None, None
            self.char_vocab_size = -1
            self.char_embeddings = None
            self.char_embed_dim = -1

        # build bert vocabulary
        if self.use_bert:
            self.bert_vocab = {}
            with codecs.open(bert_vocab_file, 'r', 'utf8') as reader:
                for line in reader:
                    token = line.strip()
                    self.bert_vocab[token] = len(self.bert_vocab)
            self.bert_tokenizer = ChineseBertTokenizer(self.bert_vocab)

        # build label vocabulary
        self.label_vocab, self.id2label = self.build_label_vocab(self.train_labels)
        self.num_class = len(self.label_vocab)

        if self.use_bert_model and self.max_len is None:
            # max_len should be provided when use bert model!
            # We will reset max_len from train_data when max_len is not provided.
            self.max_len = get_len_from_corpus([list(a) + list(b) for a, b in zip(train_data_a,
                                                                                  train_data_b)])
            self.max_len += 3  # consider 3 more special tokens: <CLS> <SEQ> <SEQ>
        elif not self.use_word and self.use_bert and self.max_len is None:
            # max_len should be provided when use bert as input!
            # We will reset max_len from train_data when max_len is not provided.
            self.max_len = get_len_from_corpus(train_data)
            self.max_len += 2  # consider 2 more special tokens: <CLS> <SEQ>

        if self.use_bert:
            # max length is 512 for bert
            self.max_len = min(self.max_len, 512)

    def load_word_dict(self):
        if self.external_word_dict:
            for word in self.external_word_dict:
                jieba.add_word(word, freq=1000000)

    @staticmethod
    def load_label_dict(label_dict_file):
        result_dict = dict()
        if label_dict_file:
            with codecs.open(label_dict_file, encoding='utf-8') as f_label_dict:
                for line in f_label_dict:
                    line_items = line.strip().split('\t')
                    result_dict[line_items[0]] = line_items[1]
            return result_dict
        else:
            return None

    def build_label_vocab(self, labels):
        """Build label vocabulary

        Args:
            labels: list of str, the label strings
        """
        label_count = {}
        for label in labels:
            label_count[label] = label_count.get(label, 0) + 1

        # sorted by frequency, so that the label with the highest frequency will be given
        # id of 0, which is the default id for unknown labels
        sorted_label_count = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
        sorted_label_count = dict(sorted_label_count)

        label_vocab = {}
        for label in sorted_label_count:
            label_vocab[label] = len(label_vocab)

        id2label = dict((idx, label) for label, idx in label_vocab.items())

        logging.info('Build label vocabulary finished, '
                     'vocabulary size: {}'.format(len(label_vocab)))
        return label_vocab, id2label

    def prepare_input(self, data, labels=None):
        """Prepare input (features and labels) for SPM model.
        Here we not only use character embeddings (or bert embeddings) as main input, but also
        support word embeddings and other hand-crafted features embeddings as additional input.

        Args:
            data: list of text pairs, like ``[['我是中国人', ...], ['我爱中国', ...]]``
            labels: list of str, the corresponding label strings

        Returns:
            features: id matrix
            y: label id matrix (only if labels is provided)

        """
        batch_word_ids_a, batch_char_ids_a, batch_bert_ids_a, batch_bert_seg_ids_a = \
            [], [], [], []
        batch_word_ids_b, batch_char_ids_b, batch_bert_ids_b, batch_bert_seg_ids_b = \
            [], [], [], []
        batch_label_ids = []

        for i, (text_a, text_b) in enumerate(zip(data[0], data[1])):
            if self.use_bert_model:
                indices, segments = self.bert_tokenizer.encode(first=text_a,
                                                               second=text_b,
                                                               max_len=self.max_len)
                batch_bert_ids_a.append(indices)
                batch_bert_seg_ids_a.append(segments)

            elif self.use_word:
                word_text_a = jieba.lcut(text_a)
                word_text_b = jieba.lcut(text_b)
                word_ids_a = self.get_word_ids(word_text_a)
                batch_word_ids_a.append(word_ids_a)
                word_ids_b = self.get_word_ids(word_text_b)
                batch_word_ids_b.append(word_ids_b)

                if self.use_char:
                    word_text_a = [list(word) for word in word_text_a]
                    word_text_b = [list(word) for word in word_text_b]
                    char_ids_a = [[self.char_vocab.get(char, self.char_vocab[self.unk_token])
                                   for char in token] for token in word_text_a]
                    char_ids_b = [[self.char_vocab.get(char, self.char_vocab[self.unk_token])
                                   for char in token] for token in word_text_b]
                    batch_char_ids_a.append(char_ids_a)
                    batch_char_ids_b.append(char_ids_b)

            else:
                text_a = list(text_a)
                text_b = list(text_b)

                if self.use_char:
                    char_text_a = [self.cls_token] + text_a + [self.seq_token] if self.use_bert \
                        else text_a
                    char_text_b = [self.cls_token] + text_b + [self.seq_token] if self.use_bert \
                        else text_b
                    char_ids_a = [self.char_vocab.get(token, self.char_vocab[self.unk_token])
                                  for token in char_text_a]
                    batch_char_ids_a.append(char_ids_a)
                    char_ids_b = [self.char_vocab.get(token, self.char_vocab[self.unk_token])
                                  for token in char_text_b]
                    batch_char_ids_b.append(char_ids_b)

                if self.use_bert:
                    indices_a, segments_a = self.bert_tokenizer.encode(first=''.join(text_a),
                                                                       max_len=self.max_len)
                    batch_bert_ids_a.append(indices_a)
                    batch_bert_seg_ids_a.append(segments_a)

                    indices_b, segments_b = self.bert_tokenizer.encode(first=''.join(text_b),
                                                                       max_len=self.max_len)
                    batch_bert_ids_b.append(indices_b)
                    batch_bert_seg_ids_b.append(segments_b)

            if labels is not None:
                label_ids = self.label_vocab.get(labels[i], self.get_unk_label_id())
                label_ids = to_categorical(label_ids, self.num_class).astype(int)
                batch_label_ids.append(label_ids)

        features_a, features_b = [], []
        if self.use_bert_model:
            features_a.append(self.pad_sequence(batch_bert_ids_a))
            features_a.append(self.pad_sequence(batch_bert_seg_ids_a))

        elif self.use_word:
            features_a.append(self.pad_sequence(batch_word_ids_a))
            features_b.append(self.pad_sequence(batch_word_ids_b))
            if self.use_char:
                features_a.append(pad_sequences_2d(batch_char_ids_a,
                                                   max_len_1=self.max_len,
                                                   max_len_2=self.max_word_len,
                                                   padding=self.padding_mode,
                                                   truncating=self.truncating_mode))
                features_b.append(pad_sequences_2d(batch_char_ids_b,
                                                   max_len_1=self.max_len,
                                                   max_len_2=self.max_word_len,
                                                   padding=self.padding_mode,
                                                   truncating=self.truncating_mode))

        else:
            if self.use_char:
                features_a.append(self.pad_sequence(batch_char_ids_a))
                features_b.append(self.pad_sequence(batch_char_ids_b))
            if self.use_bert:
                features_a.append(self.pad_sequence(batch_bert_ids_a))
                features_b.append(self.pad_sequence(batch_bert_ids_b))
                features_a.append(self.pad_sequence(batch_bert_seg_ids_a))
                features_b.append(self.pad_sequence(batch_bert_seg_ids_b))

        if len(features_a) == 1:
            features = [features_a[0], features_b[0]]
        else:
            features = features_a + features_b

        if not batch_label_ids:
            return features, None
        else:
            y = np.asarray(batch_label_ids)
            return features, y

    def get_word_ids(self, word_cut):
        """Given a word-level tokenized text, return the corresponding word ids.

        Args:
            word_cut: list of str, like ['我', '是'. '中国人']
            unk_idx: the index of words that do not appear in vocabulary, we usually set it to 1

        Returns: list of int, id sequence

        """
        word_ids = []
        for word in word_cut:
            word_ids.append(self.word_vocab.get(word, self.word_vocab[self.unk_token]))
        return word_ids

    def label_decode(self, pred_probs, label_dict=None):
        pred_ids = np.argmax(pred_probs, axis=-1)
        pred_labels = [self.id2label[pred_id] for pred_id in pred_ids]
        if label_dict:
            pred_labels = [label_dict[raw_label] for raw_label in pred_labels]
        return pred_labels

    def get_unk_label_id(self):
        """return a default id for label that does not exist in the label vocab

        Args:
            label: str

        Returns: int

        """
        if 'O' in self.label_vocab:
            return self.label_vocab['O']
        elif 'o' in self.label_vocab:
            return self.label_vocab['o']
        else:
            return 0

    def save(self, preprocessor_file):
        pickle.dump(self, open(preprocessor_file, 'wb'))

    @classmethod
    def load(cls, preprocessor_file):
        p = pickle.load(open(preprocessor_file, 'rb'))
        p.load_word_dict()  # reload external word dict into jieba
        return p
