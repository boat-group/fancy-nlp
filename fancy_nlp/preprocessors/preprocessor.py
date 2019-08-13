# -*- coding: utf-8 -*-

from keras.preprocessing.sequence import pad_sequences


class Preprocessor(object):
    """Base preprocessor.
    """

    def __init__(self,
                 max_len=50,
                 truncating_mode='post'):
        self.max_len = max_len
        self.truncating_mode = truncating_mode

    def load_corpus(self):
        """Load corpus.
        """
        raise NotImplementedError

    def get_vocabulary(self):
        """Get vocabulary list with frequency.
        """
        raise NotImplementedError

    def get_vocabulary2id(self):
        """Get a dictionary that key is `token`, and value is `token_id`.
        """
        raise NotImplementedError

    def get_id2vocabulary(self):
        """Get a dictionary that key is `token_id`, and value is `token`.
        """
        raise NotImplementedError

    def build_id_sequence(self):
        """Given a list, each item is a token list, return the corresponding id sequence.
        """
        raise NotImplementedError

    def pad_sequence(self, sequence_list):
        """Given a list, each item is a id sequence, return the padded sequence
        """
        return pad_sequences(sequence_list, maxlen=self.max_len, truncating=self.truncating_mode)
