# -*- coding: utf-8 -*-

import codecs

import numpy as np
import jieba
from absl import logging
from keras.utils.np_utils import to_categorical
from keras_bert import Tokenizer

from fancy_nlp.preprocessors.preprocessor import Preprocessor
from fancy_nlp.utils.other import pad_sequences_2d, get_most_len


class NERPreprocessor(Preprocessor):
    """NER preprocessor.
    """
    def __init__(self, train_data, train_labels, min_count=3, use_char=True, use_bert=False,
                 use_word=False, external_word_dict=None, bert_vocab_file=None,
                 char_embed_type=None, char_embed_dim=300, word_embed_type=None, word_embed_dim=300,
                 max_len=None, padding_mode='post',
                 truncating_mode='post'):
        """

        Args:
            train_data: a list of tokenized (in char level) texts
            train_labels: list of list, train_data's labels
            min_count: int, token of which frequency is lower than min_count will be ignored
            use_char：whether to use char embedding as input
            use_bert: whether to use bert embedding as input
            use_word: whether to use word embedding as additional input
            external_word_dict: external word dictionary, only apply when use_word is True
            bert_vocab_file: vocabulary file of pre-trained bert model, only apply when use_bert is
                             True
            char_embed_type: str, can be a pre-trained embedding filename or pre-trained embedding
                             methods (word2vec, glove, fastext)
            char_embed_dim: dimensionality of char embedding
            word_embed_type: same as char_embed_type, only apply when use_word is True
            word_embed_dim: dimensionality of word embedding
            max_len: int, max sequence len
            padding_mode:
            truncating_mode:
        """
        super(NERPreprocessor, self).__init__(max_len, padding_mode, truncating_mode)

        self.train_data = train_data
        self.train_labels = train_labels
        self.min_count = min_count
        self.use_char = use_char
        self.use_bert = use_bert
        self.use_word = use_word
        self.char_embed_type = char_embed_type
        self.word_embed_type = word_embed_type

        assert self.use_char or self.use_bert, "must use char or bert embedding as main input"
        special_token = 'bert' if self.use_bert else 'standard'

        # build char vocabulary and char embedding
        if self.use_char:
            self.char_vocab_count, self.char_vocab, self.id2char = \
                self.build_vocab(self.train_data, self.min_count, special_token)
            self.char_vocab_size = len(self.char_vocab)
            self.char_embeddings = self.build_embedding(char_embed_type, self.char_vocab,
                                                        self.train_data, char_embed_dim,
                                                        special_token)
            if self.char_embeddings is not None:
                self.char_embed_dim = self.char_embeddings.shape[1]
            else:
                self.char_embed_dim = char_embed_dim
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
            self.bert_tokenizer = Tokenizer(self.bert_vocab)

        # build word vocabulary and word embedding
        if self.use_word:
            if external_word_dict:
                for word in external_word_dict:
                    jieba.add_word(word, freq=1000000)
            untokenized_texts = [''.join(text) for text in self.train_data]
            word_corpus = self.build_corpus(untokenized_texts, cut_func=lambda x: jieba.lcut(x))
            self.word_vocab_count, self.word_vocab, self.id2word = \
                self.build_vocab(word_corpus, self.min_count, special_token)
            self.word_vocab_size = len(self.word_vocab)
            self.word_embeddings = self.build_embedding(word_embed_type, self.word_vocab,
                                                        self.train_data, word_embed_dim,
                                                        special_token)
            if self.word_embeddings is not None:
                self.word_embed_dim = self.word_embeddings.shape[1]
            else:
                self.word_embed_dim = word_embed_dim
        else:
            self.word_vocab_count, self.word_vocab, self.id2word = None, None, None
            self.word_vocab_size = -1
            self.word_embeddings = None
            self.word_embed_dim = -1

        # build label vocabulary
        self.label_vocab, self.id2label = self.build_label_vocab(self.train_labels)
        self.num_class = len(self.label_vocab)

        if self.use_bert and self.max_len is None:
            # max_len must be provided when use bert as input!
            # We will reset max_len from train_data when max_len is not provided.
            self.max_len = get_most_len(self.train_data)

    def build_label_vocab(self, labels):
        """Build label vocabulary

        Args:
            labels: list of list of str, the label strings
        """
        if self.use_bert:
            label_vocab = {self.cls_token: 0,
                           self.seq_token: 1}
        else:
            label_vocab = {}

        label_count = {}
        for sequence in labels:
            for label in sequence:
                label_count[label] = label_count.get(label, 0) + 1

        for label in label_count:
            label_vocab[label] = len(label_vocab)
        id2label = dict((idx, label) for label, idx in label_vocab.items())

        logging.info('Build label vocabulary finished, '
                     'vocabulary size: {}'.format(len(label_vocab)))
        return label_vocab, id2label

    def prepare_input(self, data, labels=None):
        """Prepare input (features and labels) for NER model.
        Here we not only use character embeddings (or bert embeddings) as main input, but also
        support word embeddings and other hand-crafted features embeddings as additional input.

        Args:
            data: list of tokenized (in char level) texts, like ``[['我', '是', '中', '国', '人']]``
            labels: list of list of str, the corresponding label strings

        Returns:
            features: id matrix
            y: label id matrix (only if labels is provided)

        """
        batch_char_ids, batch_bert_ids, batch_bert_seg_ids, batch_word_ids = [], [], [], []
        batch_label_ids = []
        for i, char_text in enumerate(data):
            if self.use_char:
                if self.use_bert:
                    char_text = [self.cls_token] + char_text + [self.seq_token]
                char_ids = [self.char_vocab.get(token, self.char_vocab[self.unk_token])
                            for token in char_text]
                batch_char_ids.append(char_ids)

            if self.use_bert:
                # to avoid non-chinese token being seg into subtokens
                text_with_space = ' '.join(char_text)
                indices, segments = self.bert_tokenizer.encode(first=text_with_space,
                                                               max_len=self.max_len)
                batch_bert_ids.append(indices)
                batch_bert_seg_ids.append(segments)

            if self.use_word:
                word_text = jieba.lcut(''.join(char_text))
                word_ids = self.get_word_ids(word_text)
                batch_word_ids.append(word_ids)

            if labels is not None:
                if self.use_bert:
                    label_str = [self.cls_token] + labels[i] + [self.seq_token]
                else:
                    label_str = labels[i]
                label_ids = [self.label_vocab[l] for l in label_str]
                label_ids = to_categorical(label_ids, self.num_class).astype(int)
                batch_label_ids.append(label_ids)

        features = []
        if self.use_char:
            features.append(self.pad_sequence(batch_char_ids))
        if self.use_bert:
            features.append(self.pad_sequence(batch_bert_ids))
            features.append(self.pad_sequence(batch_bert_seg_ids))
        if self.use_word:
            features.append(self.pad_sequence(batch_word_ids))

        if len(features) == 1:
            features = features[0]

        if not batch_label_ids:
            return features, None
        else:
            y = pad_sequences_2d(batch_label_ids, max_len_1=self.max_len, max_len_2=self.num_class,
                                 padding=self.padding_mode, truncating=self.truncating_mode)
            return features, y

    def get_word_ids(self, word_cut):
        """Given a word-level tokenized text, return the corresponding word ids in char-level
           sequence. We add the same word id to each character in the word.

        Args:
            word_cut: list of str, like ['我', '是'. '中国人']
            unk_idx: the index of words that do not appear in vocabulary, we usually set it to 1

        Returns: list of int, id sequence

        """
        word_ids = []
        for word in word_cut:
            for _ in word:
                word_ids.append(self.word_vocab.get(word, self.word_vocab[self.unk_token]))
        if self.use_bert:
            word_ids = [self.word_vocab[self.cls_token]] + word_ids + \
                       [self.word_vocab[self.seq_token]]
        return word_ids

    def label_decode(self, pred_probs, lengths=None):
        pred_ids = np.argmax(pred_probs, axis=-1)
        pred_labels = [[self.id2label[label_id] for label_id in ids] for ids in pred_ids]
        if lengths is not None:
            pred_labels = [labels[:length] for labels, length in zip(pred_labels, lengths)]
        return pred_labels
