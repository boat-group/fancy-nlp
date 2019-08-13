# -*- coding: utf-8 -*-

from fancy_nlp.preprocessors.preprocessor import Preprocessor


class NERPreprocessor(Preprocessor):
    """NER preprocessor.
    """
    def __init__(self):
        super(NERPreprocessor, self).__init__()
        pass

    def load_corpus(self):
        """Load corpus.
        """
        pass

    def get_vocabulary(self):
        """Get vocabulary list with frequency.
        """
        pass

    def get_vocabulary2id(self):
        """Get a dictionary that key is `token`, and value is `token_id`.
        """
        pass

    def get_id2vocabulary(self):
        """Get a dictionary that key is `token_id`, and value is `token`.
        """
        pass

    def build_id_sequence(self):
        """Given a list, each item is a token list, return the corresponding id sequence.
        """
        pass
