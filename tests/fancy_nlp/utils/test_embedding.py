# -*- coding: utf-8 -*-

import os
import numpy as np
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.utils import train_w2v, train_fasttext


class TestEmbedding:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')

    @classmethod
    def setup_class(self):
        self.test_corpus, _ = load_ner_data_and_labels(self.test_file)
        self.test_vocab = dict((token, i+2) for i, token in enumerate(set(self.test_corpus[0])))

    def test_train_w2v(self):
        emb = train_w2v(self.test_corpus, self.test_vocab, embedding_dim=10)
        assert emb.shape[0] == len(self.test_vocab) + 2 and emb.shape[1] == 10
        assert not np.any(emb[0])

    def test_train_fasttext(self):
        emb = train_fasttext(self.test_corpus, self.test_vocab, embedding_dim=10)
        assert emb.shape[0] == len(self.test_vocab) + 2 and emb.shape[1] == 10
        assert not np.any(emb[0])

