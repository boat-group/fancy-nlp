# -*- coding: utf-8 -*-

import codecs

from sklearn.model_selection import train_test_split


def load_ner_data_and_labels(filename, split=False, split_size=0.1, seed=42):
    """Load ner data and label from a file.

    The file should follow CoNLL format:
    Each line is a token and its label separated by tab, or a blank line indicating the end of
    a sentence.

    Args:
        filename: str, path to ner file
        split: bool, whether to split into train and test subsets
        split_size: float, the proportion of test subset, between 0.0 and 1.0
        seed: int, random seed

    Returns: If split: tuple(list, list, list, list), train data and labels as well as test data and
             labels.
             Otherwise: tuple(list, list), data and labels

    """
    with codecs.open(filename, 'r', encoding='utf8') as reader:
        token_seqs, label_seqs = [], []
        tokens, labels = [], []
        for line in reader:
            line = line.rstrip()
            if line:
                line_split = line.split('\t')
                if len(line_split) == 2:
                    token, label = line_split
                    tokens.append(token)
                    labels.append(label)
                else:
                    raise Exception('Format Error! Input file should follow CoNLL format.')
            else:
                if tokens:
                    token_seqs.append(tokens)
                    label_seqs.append(labels)
                    tokens, labels = [], []

        if tokens:  # in case there's no blank line at the end of the file
            token_seqs.append(tokens)
            label_seqs.append(labels)

    if split:
        x_train, x_test, y_train, y_test = train_test_split(token_seqs, label_seqs,
                                                            test_size=split_size, random_state=seed)
        return x_train, y_train, x_test, y_test
    else:
        return token_seqs, label_seqs









