# -*- coding: utf-8 -*-

import os

from absl import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from fasttext import train_unsupervised


def load_glove_format(filename):
    """Load pre-trained embedding from a file in glove-embedding-like format

    Args:
        filename: str, file path to pre-trained embedding

    Returns:
        word_vector: dict(str, np.array), a mapping of words to embeddings;
        embeddings_dim: int, dimensionality of embedding

    """
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()

            try:
                word = line[0]
                word_vector = np.array([float(v) for v in line[1:]])
            except ValueError:
                continue

            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)

            if len(word_vector) != embeddings_dim:
                continue

            word_vectors[word] = word_vector

    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())

    return word_vectors, embeddings_dim


def filter_embeddings(trained_embedding, embedding_dim, vocabulary, zero_init_indices=0,
                      rand_init_indices=1):
    """Build word embeddings matrix from pre-trained-embeddings

    Args:
        trained_embedding: dict(str, np.array), a mapping of words to embeddings
        embedding_dim: int, dimensionality of embedding
        vocabulary: dict. a mapping of words to indices
        zero_init_indices: int or a list, the indices which use zero-initialization. These indices
                           usually represent padding token.
        rand_init_indices: int or a list, the indices which use randomly-initialization.These
                           indices usually represent other special tokens, such as "unk" token.

    Returns: np.array, a word embedding matrix.

    """
    if isinstance(zero_init_indices, int):
        zero_init_indices = [zero_init_indices]
    if isinstance(rand_init_indices, int):
        rand_init_indices = [rand_init_indices]
    special_token_cnt = len(zero_init_indices) + len(rand_init_indices)

    emb = np.zeros(shape=(len(vocabulary) + special_token_cnt, embedding_dim), dtype='float32')
    for idx in rand_init_indices:
        emb[idx] = np.random.normal(0, 0.05, embedding_dim)

    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in trained_embedding:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = trained_embedding[w]
    logging.info('Embedding matrix created, shaped: {}, not found tokens: {}'.format(emb.shape,
                                                                                     nb_unk))
    return emb


def load_pre_trained(load_filename, vocabulary=None, zero_init_indices=0, rand_init_indices=1):
    """Load pre-trained embedding and fit into vocabulary if provided

    Args:
        load_filename: str, pre-trained embedding file, in word2vec-like format or glove-like format
        vocabulary: dict. a mapping of words to indices
        zero_init_indices: int or a list, the indices which use zero-initialization. These indices
                           usually represent padding token.
        rand_init_indices: int or a list, the indices which use randomly-initialization.These
                           indices usually represent other special tokens, such as "unk" token.
    Returns: If vocabulary is None: dict(str, np.array), a mapping of words to embeddings.
             Otherwise: np.array, a word embedding matrix.

    """
    word_vectors = {}
    try:
        model = KeyedVectors.load_word2vec_format(load_filename)
        weights = model.wv.vectors
        embedding_dim = weights.shape[1]
        for k, v in model.wv.vocab.items():
            word_vectors[k] = weights[v.index, :]
    except ValueError:
        word_vectors, embedding_dim = load_glove_format(load_filename)

    if vocabulary is not None:
        emb = filter_embeddings(word_vectors, embedding_dim, vocabulary, zero_init_indices,
                                rand_init_indices)
        return emb
    else:
        logging.info('Loading Embedding from: {}, shaped: {}'.format(load_filename,
                                                                     (len(word_vectors),
                                                                      embedding_dim)))
        return word_vectors


def train_w2v(corpus, vocabulary, zero_init_indices=0, rand_init_indices=1, embedding_dim=300):
    """Use word2vec to train on corpus to obtain embedding

    Args:
        corpus: list of tokenized texts, corpus to train on
        vocabulary: dict, a mapping of words to indices
        zero_init_indices: int or a list, the indices which use zero-initialization. These indices
                           usually represent padding token.
        rand_init_indices: int or a list, the indices which use randomly-initialization.These
                           indices usually represent other special tokens, such as "unk" token.
        embedding_dim: int, dimensionality of embedding

    Returns: np.array, a word embedding matrix.

    """
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.vectors
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word_vectors = dict((w, weights[d[w], :]) for w in d)
    emb = filter_embeddings(word_vectors, embedding_dim, vocabulary, zero_init_indices,
                            rand_init_indices)
    return emb


def train_fasttext(corpus, vocabulary, zero_init_indices=0, rand_init_indices=1, embedding_dim=300):
    """Use fasttext to train on corpus to obtain embedding

        Args:
            corpus: list of tokenized texts, corpus to train on
            vocabulary: dict, a mapping of words to indices
            zero_init_indices: int or a list, the indices which use zero-initialization. These
                               indices usually represent padding token.
            rand_init_indices: int or a list, the indices which use randomly-initialization.These
                               indices usually represent other special tokens, such as "unk" token.
            embedding_dim: int, dimensionality of embedding

        Returns: np.array, a word embedding matrix.

        """
    corpus_file_path = 'fasttext_tmp_corpus.txt'
    with open(corpus_file_path, 'w', encoding='utf8')as writer:
        for sentence in corpus:
            writer.write(' '.join(sentence) + '\n')

    model = train_unsupervised(input=corpus_file_path, model='skipgram', epoch=10, minCount=1,
                               wordNgrams=3, dim=embedding_dim)
    model_vocab = model.get_words()
    word_vectors = dict((w, model.get_word_vector(w)) for w in model_vocab)
    emb = filter_embeddings(word_vectors, embedding_dim, vocabulary, zero_init_indices,
                            rand_init_indices)
    os.remove(corpus_file_path)
    return emb
