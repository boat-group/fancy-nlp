# -*- coding: utf-8 -*-

import os
import numpy as np
from fancy_nlp.utils.data_loader import load_ner_data_and_labels
from fancy_nlp.preprocessors.ner_preprocessor import NERPreprocessor


class TestNERPreprocessor:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.test_file)
        self.preprocessor = NERPreprocessor(x_train, y_train, use_word=True,
                                            char_embed_type='word2vec', max_len=50)

    def test_init(self):
        assert len(self.preprocessor.char_vocab_count) == len(self.preprocessor.char_vocab) \
            == len(self.preprocessor.id2char)
        assert list(self.preprocessor.id2char.keys())[0] == 2
        assert self.preprocessor.char_embeddings.shape[0] == len(self.preprocessor.char_vocab) + 2
        assert self.preprocessor.char_embeddings.shape[1] == 300
        assert not np.any(self.preprocessor.char_embeddings[0])

        assert len(self.preprocessor.word_vocab_count) == len(self.preprocessor.word_vocab) \
            == len(self.preprocessor.id2word)
        assert list(self.preprocessor.id2word.keys())[0] == 2
        assert self.preprocessor.word_embeddings is None

        assert len(self.preprocessor.label_vocab) == len(self.preprocessor.id2label)
        assert list(self.preprocessor.id2label.keys())[0] == 0

    def test_pad_sequence(self):
        x = [[1, 3, 5]]
        x_padded = self.preprocessor.pad_sequence(x)

        assert x_padded.shape == (1, 50)
        assert (np.array(x_padded) ==
                np.array(x[0] + [0] * (self.preprocessor.max_len - len(x[0]))).reshape(1,  -1)).any()
