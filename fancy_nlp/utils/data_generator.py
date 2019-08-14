# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import Sequence


class NERGenerator(Sequence):
    def __init__(self, token_seqs, label_seqs=None, batch_size=32, shuffle=True):
        self.token_seqs = token_seqs
        self.label_seqs = label_seqs
        self.data_size = len(self.label_seqs)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(np.ceil(self.data_size / self.batch_size))
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        pass