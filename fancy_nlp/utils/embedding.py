# -*- coding: utf-8 -*-

from typing import Dict, Union, List, Optional

from absl import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText


def load_glove_format(filename: str, embedding_dim: int) -> Dict[str, np.ndarray]:
    """Load pre-trained embedding from a file in glove-embedding-like format:
    Each line is a token and its embedding separated by blank space.

    Args:
        filename: str. File path to pre-trained embedding.
        embedding_dim: int. Dimensionality of embedding, used to validate the embedding file.
    Returns:
        word_vector: Dict(str, np.ndarray), a mapping of words to embeddings.

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


def filter_embeddings(trained_embedding: Dict[str, np.ndarray],
                      embedding_dim: int,
                      vocabulary: Dict[str, int],
                      zero_init_indices: Union[int, List[int]] = 0,
                      rand_init_indices: Union[int, List[int]] = 1) -> np.ndarray:
    """Build word embeddings matrix from pre-trained-embeddings and word vocabulary.

    Args:
        trained_embedding: Dict(str, np.ndarray). A mapping of words to pre-trained embeddings
        embedding_dim: int. Dimensionality of embedding.
        vocabulary: Dict[str, int]. A mapping of words to indices
        zero_init_indices: int or a List of int. The indices which use zero-initialization. These
            indices usually represent padding token.
        rand_init_indices: int or a List of int. The indices which use randomly-initialization.These
            indices usually represent other special tokens, such as "unk" token.

    Returns: np.ndarray, a word embedding matrix, shaped [vocab_size, embedding_dim].

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


def load_pre_trained(load_filename: str,
                     embedding_dim: Optional[int] = None,
                     vocabulary: Optional[Dict[str, int]] = None,
                     zero_init_indices: Union[int, List[int]] = 0,
                     rand_init_indices: Union[int, List[int]] = 1) \
        -> Union[Dict[str, np.ndarray], np.ndarray]:
    """Load pre-trained embedding and fit into vocabulary if provided

    Args:
        load_filename: str. Pre-trained embedding file, in word2vec-like format or glove-like format
        embedding_dim: int. Dimensionality of embeddings in the embedding file, must be provided
            when the file is in glove-like format.
        vocabulary: Dict[str, int]. A mapping of words to indices.
        zero_init_indices: int or a List of int. The indices which use zero-initialization. These
            indices usually represent padding token.
        rand_init_indices: int or a List of int. The indices which use randomly-initialization.These
            indices usually represent other special tokens, such as "unk" token.

    Returns: If vocabulary is None: Dict(str, np.ndarray), a mapping of words to embeddings.
             Otherwise: np.ndarray, a word embedding matrix.

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


def train_w2v(corpus: List[List[str]],
              vocabulary: Dict[str, int],
              zero_init_indices: Union[int, List[int]] = 0,
              rand_init_indices: Union[int, List[int]] = 1,
              embedding_dim: int = 300) -> np.ndarray:
    """Use word2vec to train on corpus to obtain embedding.

    Args:
        corpus: List of List of str. List of tokenized texts, the corpus to train on, like ``[['我',
            '是', '中', '国', '人'], ...]``.
        vocabulary: Dict[str, int']. A mapping of words to indices
        zero_init_indices: int or a List of int. The indices which use zero-initialization. These
            indices usually represent padding token.
        rand_init_indices: int or a List of int. The indices which use randomly-initialization.These
            indices usually represent other special tokens, such as "unk" token.
        embedding_dim: int. Dimensionality of embedding

    Returns: np.ndarray, a word embedding matrix, shaped [vocab_size, embedding_dim].

    """
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.vectors
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word_vectors = dict((w, weights[d[w], :]) for w in d)
    emb = filter_embeddings(word_vectors, embedding_dim, vocabulary, zero_init_indices,
                            rand_init_indices)
    return emb


def train_fasttext(corpus: List[List[str]],
                   vocabulary: Dict[str, int],
                   zero_init_indices: Union[int, List[int]] = 0,
                   rand_init_indices: Union[int, List[int]] = 1,
                   embedding_dim: int = 300) -> np.ndarray:
    """Use fasttext to train on corpus to obtain embedding

    Args:
        corpus: List of List of str. List of tokenized texts, the corpus to train on, like ``[['我',
            '是', '中', '国', '人'], ...]``.
        vocabulary: Dict[str, int']. A mapping of words to indices
        zero_init_indices: int or a List of int. The indices which use zero-initialization. These
            indices usually represent padding token.
        rand_init_indices: int or a List of int. The indices which use randomly-initialization.These
            indices usually represent other special tokens, such as "unk" token.
        embedding_dim: int. Dimensionality of embedding

    Returns: np.ndarray, a word embedding matrix, shaped [vocab_size, embedding_dim].

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
