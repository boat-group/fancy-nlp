# -*- coding: utf-8 -*-

import numpy as np

from fancy_nlp.preprocessors.ner_preprocessor import NERPreprocessor


class TestNERPreprocessor:
    def test_pad_sequence(self):
        preprocessor = NERPreprocessor()
        x = [[1, 3, 5]]
        x_padded = preprocessor.pad_sequence(x)

        assert x_padded.shape == (1, 50)
        assert (np.array(x_padded) ==
                np.array(x[0] + [0] * (preprocessor.max_len - len(x[0]))).reshape(1,  -1)).any()
