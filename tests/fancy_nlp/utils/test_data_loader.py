# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels, load_spm_data_and_labels


class TestNerDataLoader:
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


class TestSpmDataLoader:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/spm/webank/example.txt')

    def test_load_spm(self):
        x_train, y_train = load_spm_data_and_labels(self.test_file)
        assert len(x_train) == 2
        assert len(x_train[0]) == len(x_train[1])
        assert len(x_train[0]) == len(y_train)
        assert len(x_train[0]) > 0

    def test_load_spm_split1(self):
        x_train, y_train, x_test, y_test = load_spm_data_and_labels(self.test_file, split_mode=1)
        assert len(x_train[0]) == len(x_train[1])
        assert len(x_test[0]) == len(x_test[1])
        assert len(x_train[0]) == len(y_train) and len(x_test[0]) == len(y_test)
        assert len(x_train[0]) > 0 and len(x_test[0]) > 0

    def test_load_spm_split2(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test = \
            load_spm_data_and_labels(self.test_file, split_mode=2)
        assert len(x_train[0]) == len(x_train[1]) == len(y_train)
        assert len(x_valid[0]) == len(x_valid[1]) == len(y_valid)
        assert len(x_test[0]) == len(x_test[1]) == len(y_test)
        assert len(x_train[0]) > 0 and len(x_test[0]) > 0