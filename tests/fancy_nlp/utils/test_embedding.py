# -*- coding: utf-8 -*-

import os
import numpy as np
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.utils import train_w2v, train_fasttext, load_pre_trained


class TestEmbedding:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')
    embedding_file = os.path.join(os.path.dirname(__file__),
                                  '../../../data/embeddings/Tencent_ChineseEmbedding_example.txt')

    def setup_class(self):
        self.test_corpus, _ = load_ner_data_and_labels(self.test_file)
        self.test_vocab = {'<PAD>': 0, '<UNK>': 1}
        for token in set(self.test_corpus[0]):
            self.test_vocab[token] = len(self.test_vocab)

    def test_train_w2v(self):
        emb = train_w2v(self.test_corpus, self.test_vocab, embedding_dim=10)
        assert emb.shape[0] == len(self.test_vocab) and emb.shape[1] == 10
        assert not np.any(emb[0])

    def test_train_fasttext(self):
        emb = train_fasttext(self.test_corpus, self.test_vocab, embedding_dim=10)
        assert emb.shape[0] == len(self.test_vocab) and emb.shape[1] == 10
        assert not np.any(emb[0])

    def test_load_pre_trained(self):
        emb = load_pre_trained(load_filename=self.embedding_file,
                               embedding_dim=200,
                               vocabulary=self.test_vocab)
        assert emb.shape[0] == len(self.test_vocab) and emb.shape[1] == 200
        assert not np.any(emb[0])
