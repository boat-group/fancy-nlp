# -*- coding: utf-8 -*-

import numpy as np


def pad_sequences_2d(sequences, max_len_1=None, max_len_2=None, dtype='int32', padding='post',
                     truncating='post', value=0.):
    """pad sequence for [[[a, b, c, ...]]]"""
    lengths_1, lengths_2 = [], []
    for s in sequences:
        lengths_1.append(len(s))
        for t in s:
            lengths_2.append(len(t))
    if max_len_1 is None:
        max_len_1 = np.max(lengths_1)
    if max_len_2 is None:
        max_len_2 = np.max(lengths_2)

    num_samples = len(sequences)
    x = (np.ones((num_samples, max_len_1, max_len_2)) * value).astype(dtype)
    for i, s in enumerate(sequences):
        if not len(s):
            continue    # empty list was found

        if truncating == 'pre':
            s = s[-max_len_1:]
        elif truncating == 'post':
            s = s[:max_len_1]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        y = (np.ones((len(s), max_len_2)) * value).astype(dtype)
        for j, t in enumerate(s):
            if not len(t):
                continue

            if truncating == 'pre':
                trunc = t[-max_len_2:]
            elif truncating == 'post':
                trunc = t[:max_len_2]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            trunc = np.asarray(trunc, dtype=dtype)

            if padding == 'post':
                y[j, :len(trunc)] = trunc
            elif padding == 'pre':
                y[j, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)

        if padding == 'post':
            x[i, :y.shape[0], :] = y
        elif padding == 'pre':
            x[i, -y.shape[0]:, :] = y
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

    return x
