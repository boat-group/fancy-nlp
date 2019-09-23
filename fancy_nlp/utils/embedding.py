# -*- coding: utf-8 -*-

from absl import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText


def load_glove_format(filename, embedding_dim):
    """Load pre-trained embedding from a file in glove-embedding-like format

    Args:
        filename: str, file path to pre-trained embedding
        embedding_dim: int, dimensionality of embedding
    Returns:
        word_vector: dict(str, np.array), a mapping of words to embeddings;

    """
    word_vectors = {}
    with open(filename, 'r') as reader:
        for i, line in enumerate(reader):
            line = line.strip().split()
            word = line[0]
            word_vector = np.array([float(v) for v in line[1:]])

            if word_vector.shape[0] != embedding_dim:
                raise ValueError(f'Format error at line {i}! The size of word embedding dose not '
                                 f'equal {embedding_dim}')

            word_vectors[word] = word_vector

    return word_vectors


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
    emb = np.zeros(shape=(len(vocabulary), embedding_dim), dtype='float32')
    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in trained_embedding:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = trained_embedding[w]

    if isinstance(zero_init_indices, int):
        zero_init_indices = [zero_init_indices]
    if isinstance(rand_init_indices, int):
        rand_init_indices = [rand_init_indices]
    for idx in zero_init_indices:
        emb[idx] = np.zeros(embedding_dim)
    for idx in rand_init_indices:
        emb[idx] = np.random.normal(0, 0.05, embedding_dim)

    logging.info('Embedding matrix created, shaped: {}, not found tokens: {}'.format(emb.shape,
                                                                                     nb_unk))
    return emb


def load_pre_trained(load_filename, embedding_dim=None, vocabulary=None, zero_init_indices=0,
                     rand_init_indices=1):
    """Load pre-trained embedding and fit into vocabulary if provided

    Args:
        load_filename: str, pre-trained embedding file, in word2vec-like format or glove-like format
        embedding_dim: int, dimensionality of embeddings in the embedding file, must be provided when
                       the file is in glove-like format.
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
        # load word2vec-like format pre-trained embedding file
        model = KeyedVectors.load_word2vec_format(load_filename)
        weights = model.wv.vectors
        word_embed_dim = weights.shape[1]
        for k, v in model.wv.vocab.items():
            word_vectors[k] = weights[v.index, :]
    except ValueError:
        # load glove-like format pre-trained embedding file
        if embedding_dim is None:
            raise ValueError('`embedding_dim` must be provided when pre-trained embedding file is'
                             'in glove-like format!')
        word_vectors = load_glove_format(load_filename, embedding_dim)
        word_embed_dim = embedding_dim

    if vocabulary is not None:
        emb = filter_embeddings(word_vectors, word_embed_dim, vocabulary, zero_init_indices,
                                rand_init_indices)
        return emb
    else:
        logging.info('Loading Embedding from: {}, shaped: {}'.format(load_filename,
                                                                     (len(word_vectors),
                                                                      word_embed_dim)))
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
    model = FastText(size=embedding_dim, min_count=1, window=5, sg=1, word_ngrams=1)
    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

    emb = np.zeros(shape=(len(vocabulary), embedding_dim), dtype='float32')

    for w, i in vocabulary.items():
        emb[i, :] = model.wv[w]  # note that oov words can still have word vectors

    if isinstance(zero_init_indices, int):
        zero_init_indices = [zero_init_indices]
    if isinstance(rand_init_indices, int):
        rand_init_indices = [rand_init_indices]
    for idx in zero_init_indices:
        emb[idx] = np.zeros(embedding_dim)
    for idx in rand_init_indices:
        emb[idx] = np.random.normal(0, 0.05, embedding_dim)

    return emb
