# -*- coding: utf-8 -*-

from absl import logging
import jieba
from fancy_nlp.preprocessors.preprocessor import Preprocessor


class NERPreprocessor(Preprocessor):
    """NER preprocessor.
    """
    def __init__(self, train_data, train_label, min_count=3, start_index=2, use_word=False,
                 external_word_dict=None, char_embed_type=None, word_embed_type=None,
                 max_len=None, padding_mode='post',
                 truncating_mode='post'):
        """

        Args:
            train_data: a list of tokenized (in char level) sentences
            train_label: list of list, train_data's labels
            min_count:
            start_index:
            use_word: whether to use word as additional input (here we use char as main input for
                      Chinese NER)
            external_word_dict: external word dictionary
            char_embed_type: str, can be a pre-trained embedding filename or pre-trained embedding
                             methods (word2vec, glove, fastext)
            word_embed_type:
            max_len:
            padding_mode:
            truncating_mode:
        """
        self.train_data = train_data
        self.train_label = train_label
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
            self.word_embeddings = self.build_embedding(word_embed_type, self.word_vocab,
                                                        self.train_data, pad_idx=0, unk_idx=1)

        # build label vocabulary
        _, self.label_vocab, self.id2label = self.build_label_vocab(self.train_label)

    def build_label_vocab(self, labels):
        """Build label vocabulary
        """
        return self.build_vocab(labels, min_count=0, start_index=0)

