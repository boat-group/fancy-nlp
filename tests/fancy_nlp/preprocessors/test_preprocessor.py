# -*- coding: utf-8 -*-

import os
import jieba
import numpy as np
from fancy_nlp.preprocessors.preprocessor import Preprocessor


class TestPreprocessor:
    sample_texts = ['文献是朝阳区区长',
                    '拳王阿里是个传奇',
                    '杭州阿里吸引了很多人才',
                    '习近平常与特朗普通电话',
                    '南京市长江大桥']
    embedding_file = os.path.join(os.path.dirname(__file__),
                                  '../../../data/embeddings/Tencent_ChineseEmbedding_example.txt')

    def setup_class(self):
        self.preprocessor = Preprocessor(max_len=50)

    def test_build_corpus(self):
        char_corpus = self.preprocessor.build_corpus(self.sample_texts,
                                                     cut_func=lambda x: list(x))
        assert len(char_corpus) == len(self.sample_texts)
        assert ''.join(char_corpus[0]) == self.sample_texts[0]

        word_corpus = self.preprocessor.build_corpus(self.sample_texts,
                                                     cut_func=lambda x: jieba.lcut(x))
        assert len(word_corpus) == len(self.sample_texts)
        assert ''.join(word_corpus[0]) == self.sample_texts[0]

    def test_build_vocab(self):
        char_corpus = self.preprocessor.build_corpus(self.sample_texts,
                                                     cut_func=lambda x: list(x))
        char_vocab_count, char_vocab, id2char = self.preprocessor.build_vocab(
            char_corpus, min_count=1)
        assert len(char_vocab_count) + 2 == len(char_vocab) == len(id2char)
        assert list(id2char.keys())[0] == 0

    def test_build_embedding(self):
        char_corpus = self.preprocessor.build_corpus(self.sample_texts,
                                                     cut_func=lambda x: list(x))
        _, char_vocab, _ = self.preprocessor.build_vocab(char_corpus, min_count=1)
        emb = self.preprocessor.build_embedding(embed_type=None, vocab=char_vocab)
        assert emb is None

        emb = self.preprocessor.build_embedding(embed_type='word2vec', vocab=char_vocab,
                                                corpus=char_corpus)
        assert emb.shape[0] == len(char_vocab) and emb.shape[1] == 300
        assert not np.any(emb[0])

        emb = self.preprocessor.build_embedding(embed_type='fasttext', vocab=char_vocab,
                                                corpus=char_corpus, embedding_dim=20)
        assert emb.shape[0] == len(char_vocab) and emb.shape[1] == 300
        assert not np.any(emb[0])

        emb = self.preprocessor.build_embedding(embed_type=self.embedding_file,
                                                embedding_dim=200,
                                                vocab=char_vocab,
                                                corpus=char_corpus)
        assert emb.shape[0] == len(char_vocab) and emb.shape[1] == 200
        assert not np.any(emb[0])

    def test_build_id_sequence(self):
        char_corpus = self.preprocessor.build_corpus(self.sample_texts,
                                                     cut_func=lambda x: list(x))
        _, char_vocab, _ = self.preprocessor.build_vocab(char_corpus, min_count=1)
        sample_text = list('文献是朝阳区区长吗？')
        id_sequence = self.preprocessor.build_id_sequence(sample_text, char_vocab)
        assert len(id_sequence) == len(sample_text)
        assert id_sequence[-1] == 1

    def test_build_id_matrix(self):
        sample_texts = [list('文献是朝阳区区长吗？'), list('拳王阿里是个传奇啊！')]
        char_corpus = self.preprocessor.build_corpus(self.sample_texts,
                                                     cut_func=lambda x: list(x))
        _, char_vocab, _ = self.preprocessor.build_vocab(char_corpus, min_count=1)
        id_matrix = self.preprocessor.build_id_matrix(sample_texts, char_vocab)
        assert len(id_matrix) == len(sample_texts)
        assert len(id_matrix[-1]) == len(sample_texts[-1])
        assert id_matrix[-1][-1] == 1

    def test_pad_sequence(self):
        x = [[1, 3, 5]]
        x_padded = self.preprocessor.pad_sequence(x)

        assert x_padded.shape == (1, 50)
        assert (np.array(x_padded) ==
                np.array(x[0] + [0] * (self.preprocessor.max_len - len(x[0]))).reshape(1, -1)).any()
