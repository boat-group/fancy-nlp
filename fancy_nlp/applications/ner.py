# -*- coding: utf-8 -*-

from absl import logging
import pickle
from fancy_nlp.preprocessors import NERPreprocessor
from fancy_nlp.models.ner import *
from fancy_nlp.trainers import NERTrainer
from fancy_nlp.predictors import NERPredictor
from fancy_nlp.utils import load_keras_model, save_keras_model


class NER(object):
    """NER application"""

    def __init__(self,
                 checkpoint_dir,
                 ner_model_type,
                 char_embed_type='word2vec',
                 char_embed_dim=300,
                 char_embed_trainable=False,
                 use_word=False,
                 external_word_dict=None,
                 word_embed_type='word2vec',
                 word_embed_dim=300,
                 word_embed_trainable=False,
                 use_crf=True,
                 optimizer='adam',
                 **kwargs):
        """

        Args:
            checkpoint_dir: str, dir to save ner model and ner preprocessor
            ner_model_type: str, which ner model to use
            char_embed_type: str, can be a pre-trained embedding filename or pre-trained embedding
                             methods (word2vec, glove, fastext)
            char_embed_dim: int, dimensionality of char embedding
            char_embed_trainable: boolean, whether to update char embedding during training
            use_word: boolean, whether to use word as additional input
            external_word_dict: external word dictionary
            word_embed_dim: similar as 'char_embed_dim'
            word_embed_type: similar as 'char_embed_type'
            word_embed_trainable: similar as 'char_embed_trainable'
            use_crf: boolean, whether to use crf layer
            optimizer: str, optimizer to use during training
            **kwargs: other argument for building ner model, such as "rnn_units", "fc_dim" etc
        """
        self.checkpoint_dir = checkpoint_dir
        assert isinstance(self.checkpoint_dir, str)
        self.ner_model_type = ner_model_type
        self.char_embed_type = char_embed_type
        self.char_embed_dim = char_embed_dim
        self.char_embed_trainable = char_embed_trainable
        self.use_word = use_word
        self.external_word_dict = external_word_dict
        self.word_embed_dim = word_embed_dim
        self.word_embed_type = word_embed_type
        self.word_embed_trainable = word_embed_trainable
        self.use_crf = use_crf
        self.optimizer = optimizer
        self.kwargs = kwargs

        self.preprocessor = None
        self.model = None
        self.trainer = None
        self.predictor = None

    def get_model(self, num_class, char_vocab_size, char_embeddings=None, word_embeddings=None,
                  word_vocab_size=-1):
        if self.ner_model_type == 'bilstm':
            return BiLSTMNER(num_class=num_class,
                             char_embeddings=char_embeddings,
                             char_vocab_size=char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=self.char_embed_trainable,
                             use_word=self.use_word,
                             word_embeddings=word_embeddings,
                             word_vocab_size=word_vocab_size,
                             word_embed_dim=self.word_embed_dim,
                             word_embed_trainable=self.word_embed_trainable,
                             use_crf=self.use_crf,
                             optimizer=self.optimizer,
                             checkpoint_dir=self.checkpoint_dir,
                             **self.kwargs)
        elif self.ner_model_type == 'bilstm_cnn':
            return BiLSTMCNNNER(num_class=num_class,
                                char_embeddings=char_embeddings,
                                char_vocab_size=char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=self.char_embed_trainable,
                                use_word=self.use_word,
                                word_embeddings=word_embeddings,
                                word_vocab_size=word_vocab_size,
                                word_embed_dim=self.word_embed_dim,
                                word_embed_trainable=self.word_embed_trainable,
                                use_crf=self.use_crf,
                                optimizer=self.optimizer,
                                checkpoint_dir=self.checkpoint_dir,
                                **self.kwargs)
        elif self.ner_model_type == 'bigru':
            return BiGRUNER(num_class=num_class,
                            char_embeddings=char_embeddings,
                            char_vocab_size=char_vocab_size,
                            char_embed_dim=self.char_embed_dim,
                            char_embed_trainable=self.char_embed_trainable,
                            use_word=self.use_word,
                            word_embeddings=word_embeddings,
                            word_vocab_size=word_vocab_size,
                            word_embed_dim=self.word_embed_dim,
                            word_embed_trainable=self.word_embed_trainable,
                            use_crf=self.use_crf,
                            optimizer=self.optimizer,
                            checkpoint_dir=self.checkpoint_dir,
                            **self.kwargs)
        elif self.ner_model_type == 'bilstm_cnn':
            return BiGRUCNNNER(num_class=num_class,
                               char_embeddings=char_embeddings,
                               char_vocab_size=char_vocab_size,
                               char_embed_dim=self.char_embed_dim,
                               char_embed_trainable=self.char_embed_trainable,
                               use_word=self.use_word,
                               word_embeddings=word_embeddings,
                               word_vocab_size=word_vocab_size,
                               word_embed_dim=self.word_embed_dim,
                               word_embed_trainable=self.word_embed_trainable,
                               use_crf=self.use_crf,
                               optimizer=self.optimizer,
                               checkpoint_dir=self.checkpoint_dir,
                               **self.kwargs)
        else:
            raise ValueError('ner_mode_type not understood: {}'.format(self.ner_model_type))

    def fit(self, train_data, train_labels, valid_data=None, valid_labels=None, batch_size=32,
            epochs=50, callbacks=None, load_swa_model=False, shuffle=True):
        """Train ner model using provided data

        Args:
            train_data: list of tokenized (in char level) texts for training,
                        like ``[['我', '是', '中', '国', '人']]``
            train_labels: labels string of train_data
            valid_data: list of tokenized (in char level) texts for training,
            valid_labels: labels string of valid data
            batch_size:
            epochs:
            callbacks: ist of str, each item indicate the callback to apply during training.
                       For example, 'earlystopping' means using 'EarlyStopping' callback.
            shuffle:

        """
        self.preprocessor = NERPreprocessor(train_data=train_data,
                                            train_labels=train_labels,
                                            use_word=self.use_word,
                                            external_word_dict=self.external_word_dict,
                                            char_embed_type=self.char_embed_type,
                                            word_embed_type=self.word_embed_type)
        if self.use_word:
            self.model = self.get_model(num_class=self.preprocessor.num_class,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embeddings=self.preprocessor.char_embeddings,
                                        word_embeddings=self.preprocessor.word_embeddings,
                                        word_vocab_size=self.preprocessor.word_vocab_size)
        else:
            self.model = self.get_model(num_class=self.preprocessor.num_class,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embeddings=self.preprocessor.char_embeddings)

        self.trainer = NERTrainer(self.model, self.preprocessor)
        self.trainer.train_generator(train_data, train_labels, valid_data, valid_labels,
                                     batch_size, epochs, callbacks, shuffle)

        if load_swa_model and callbacks is not None and 'swa' in callbacks:
            self.model.load_swa_model()

        if valid_data is not None and valid_labels is not None:
            self.score(valid_data, valid_labels)

        self.predictor = NERPredictor(self.model, self.preprocessor)

    def score(self, valid_data, valid_labels):
        """Return the f1 score of the model over validation data

        Args:
            valid_data: list of tokenized texts
            valid_labels: list of label strings

        Returns:

        """
        if self.trainer:
            return self.trainer.evaluate(valid_data, valid_labels)
        else:
            logging.fatal('Trainer is None! Call fit() or load() to get trainer.')

    def predict(self, test_text):
        """Return prediction of the model for test data

        Args:
            test_text: untokenized text or tokenized (in char level) text

        Returns:

        """
        if self.predictor:
            return self.predictor.tag(test_text)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def predict_batch(self, test_texts):
        """Return predictions of the model for test data

        Args:
            test_texts: list of untokenized texts or tokenized (in char level) texts

        Returns:

        """
        if self.predictor:
            return self.predictor.tag_batch(test_texts)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def analyze(self, text):
        """Analyze text and return pretty format.

        Args:
            text: untokenized text or tokenized (in char level) text
        Returns:

        """
        if self.predictor:
            return self.predictor.pretty_tag(text)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def analyze_batch(self, texts):
        """Analyze batch of texts and return pretty format.

        Args:
            texts: untokenized texts or tokenized (in char level) texts
        Returns:

        """
        if self.predictor:
            return self.predictor.pretty_tag_batch(texts)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def save_preprocessor(self, preprocessor_file):
        with open(preprocessor_file, 'wb') as writer:
            pickle.dump(self.preprocessor, writer)

    @staticmethod
    def load_preprocessor(preprocessor_file):
        with open(preprocessor_file, 'rb') as reader:
            return pickle.load(reader)

    def save(self, json_file, weights_file, preprocessor_file):
        self.save_preprocessor(preprocessor_file)
        save_keras_model(self.model.model, json_file, weights_file)

    def load(self, json_file, weights_file, preprocessor_file):
        self.preprocessor = self.load_preprocessor(preprocessor_file)

        if self.use_word:
            self.model = self.get_model(num_class=self.preprocessor.num_class,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embeddings=self.preprocessor.char_embeddings,
                                        word_embeddings=self.preprocessor.word_embeddings,
                                        word_vocab_size=self.preprocessor.word_vocab_size)
        else:
            self.model = self.get_model(num_class=self.preprocessor.num_class,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embeddings=self.preprocessor.char_embeddings)

        self.model.model = load_keras_model(json_file, weights_file,
                                            custom_objects={'CRF': CRF} if self.use_crf else None)
        self.trainer = NERTrainer(self.model, self.preprocessor)
        self.predictor = NERPredictor(self.model, self.preprocessor)
