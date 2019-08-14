# -*- coding: utf-8 -*-

import numpy as np
from fancy_nlp.utils.other import pad_sequences_2d


class TestOther:
    def test_pad_sequence_2d(self):
        test_case = [[[1, 2, 4], [1, 2], [3]],
                     [[2, 4], [1, 0, 2]],
                     [[1, 2, 3, 4, 5]]]
        expected = [[[1, 2, 4], [0, 1, 2]],
                    [[0, 2, 4], [1, 0, 2]],
                    [[0, 0, 0], [1, 2, 3]]]
        result = pad_sequences_2d(test_case, max_len_1=2, max_len_2=3, padding='pre',
                                  truncating='post')
        np.testing.assert_equal(result, expected)

        test_case = [[[1, 2, 3, 4], [1, 2], [1], [1, 2, 3]],
                     [[1, 2, 3, 4, 5], [1, 2], [1, 2, 3, 4]]]
        expected = [[[1, 2, 3, 4, 0], [1, 2, 0, 0, 0], [1, 0, 0, 0, 0], [1, 2, 3, 0, 0]],
                    [[1, 2, 3, 4, 5], [1, 2, 0, 0, 0], [1, 2, 3, 4, 0], [0, 0, 0, 0, 0]]]
        result = pad_sequences_2d(test_case)
        np.testing.assert_equal(result, expected)
