# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels


class TestDataLoader:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')

    def test_load_ner(self):
        x_train, y_train = load_ner_data_and_labels(self.test_file)
        assert len(x_train) == len(y_train)
        assert len(x_train) > 0
        assert len(x_train[0]) == len(y_train[0])
        assert len(x_train[0]) > 0
        assert x_train[:5] != y_train[:5]

    def test_load_ner_split(self):
        x_train, y_train, x_test, y_test = load_ner_data_and_labels(self.test_file, split=True)
        assert len(x_train) == len(y_train) and len(x_test) == len(y_test)
        assert len(x_train) > 0 and len(x_train) > 0
        assert len(x_train[0]) == len(y_train[0]) and len(x_test[0]) == len(y_test[0])
        assert len(x_train[0]) > 0 and len(x_test[0]) > 0
        assert x_train[:5] != y_train[:5] and x_test[:5] != y_test[:5]
        assert x_train[:5] != x_test[:5] and y_train[:5] != y_test[:5]






