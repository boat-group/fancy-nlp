# -*- coding: utf-8 -*-

import numpy as np
import jieba
from keras.utils.np_utils import to_categorical
from fancy_nlp.preprocessors.preprocessor import Preprocessor
from fancy_nlp.utils.other import pad_sequences_2d


class NERPreprocessor(Preprocessor):
    """NER preprocessor.
    """
    def __init__(self, train_data, train_labels, min_count=3, start_index=2, use_word=False,
                 external_word_dict=None, char_embed_type=None, word_embed_type=None,
                 max_len=None, padding_mode='post',
                 truncating_mode='post'):
        """

        Args:
            train_data: a list of tokenized (in char level) texts
            train_labels: list of list, train_data's labels
            min_count: int, token whose frequency is lower than min_count will be ignored
            start_index: the starting index of tokens
            use_word: whether to use word as additional input (here we use char as main input for
                      Chinese NER)
            external_word_dict: external word dictionary
            char_embed_type: str, can be a pre-trained embedding filename or pre-trained embedding
                             methods (word2vec, glove, fastext)
            word_embed_type: same as char_embed_type
            max_len: int, max sequence len
            padding_mode:
            truncating_mode:
        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.min_count = min_count
        self.start_index = start_index
        self.use_word = use_word
        self.external_word_dict = external_word_dict

        super(NERPreprocessor, self).__init__(max_len, padding_mode, truncating_mode)

        # build character vocabulary
        self.char_vocab_count, self.char_vocab, self.id2char = self.build_vocab(self.train_data,
                                                                                self.min_count,
                                                                                self.start_index)
        self.char_embeddings = self.build_embedding(char_embed_type, self.char_vocab,
                                                    self.train_data, pad_idx=0, unk_idx=1)
        self.char_vocab_size = len(self.char_vocab) + 2

        # build word vocabulary
        if self.use_word:
            if self.external_word_dict:
                for word in self.external_word_dict:
                    jieba.add_word(word, freq=1000000)
            untokenized_texts = [''.join(text) for text in self.train_data]
            word_corpus = self.build_corpus(untokenized_texts, cut_func=lambda x: jieba.lcut(x))
            self.word_vocab_count, self.word_vocab, self.id2word = self.build_vocab(word_corpus,
                                                                                    self.min_count,
                                                                                    self.start_index)
            self.word_vocab_size = len(self.word_vocab) + 2
            self.word_embeddings = self.build_embedding(word_embed_type, self.word_vocab,
                                                        self.train_data, pad_idx=0, unk_idx=1)

        # build label vocabulary
        _, self.label_vocab, self.id2label = self.build_label_vocab(self.train_labels)
        self.num_class = len(self.label_vocab)

    def build_label_vocab(self, labels):
        """Build label vocabulary

        Args:
            labels: list of list of str, the label strings
        """
        return self.build_vocab(labels, min_count=0, start_index=0)

    def prepare_input(self, data, labels=None):
        """Prepare input (features and labels) for NER model.
        Here we not only use character embeddings as main input, but also support word embeddings
        and other hand-crafted features embeddings as additional input as well.

        Args:
            data: list of tokenized (in char level) texts, like ``[['我', '是', '中', '国', '人']]``
            labels: list of list of str, the corresponding label strings

        Returns:
            features: id matrix
            y: label id matrix (only if label is not None)

        """
        batch_char_ids, batch_word_ids, batch_label_ids = [], [], []
        for i, char_text in enumerate(data):
            char_ids = self.build_id_sequence(char_text, self.char_vocab)
            batch_char_ids.append(char_ids)

            if self.use_word:
                word_text = jieba.lcut(''.join(char_text))
                word_ids = self.get_word_ids(word_text)
                batch_word_ids.append(word_ids)

            if labels is not None:
                label_ids = self.build_id_sequence(labels[i], self.label_vocab, unk_idx=-1)
                label_ids = to_categorical(label_ids, self.num_class).astype(int)
                batch_label_ids.append(label_ids)

        features = [self.pad_sequence(batch_char_ids)]
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

    def get_word_ids(self, word_cut, unk_idx=1):
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
                word_ids.append(self.word_vocab.get(word, unk_idx))
        return word_ids

    def label_decode(self, pred_probs, lengths=None):
        pred_ids = np.argmax(pred_probs, axis=-1)
        pred_labels = [[self.id2label[label_id] for label_id in ids] for ids in pred_ids]
        if lengths is not None:
            pred_labels = [labels[:length] for labels, length in zip(pred_labels, lengths)]
        return pred_labels

