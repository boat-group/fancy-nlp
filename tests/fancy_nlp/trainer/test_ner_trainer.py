# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.models.ner import BiLSTMCNNNER
from fancy_nlp.preprocessors import NERPreprocessor
from fancy_nlp.trainers import NERTrainer


class TestNERTrainer:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')

    def setup_class(self):
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = \
            load_ner_data_and_labels(self.test_file, split=True)
        self.preprocessor = NERPreprocessor(self.train_data+self.valid_data,
                                            self.valid_data+self.valid_labels, use_word=True,
                                            char_embed_type='word2vec', word_embed_type='word2vec')
        self.num_class = self.preprocessor.num_class
        self.char_embeddings = self.preprocessor.char_embeddings
        self.char_vocab_size = self.char_embeddings.shape[0]
        self.char_embed_dim = self.char_embeddings.shape[1]

        self.word_embeddings = self.preprocessor.word_embeddings
        self.word_vocab_size = self.word_embeddings.shape[0]
        self.word_embed_dim = self.word_embeddings.shape[1]
        self.checkpoint_dir = os.path.dirname(__file__)

        self.ner_model = BiLSTMCNNNER(self.num_class, self.char_embeddings, self.char_vocab_size,
                                      self.char_embed_dim, False, use_word=True,
                                      word_embeddings=self.word_embeddings,
                                      word_vocab_size=self.word_vocab_size,
                                      word_embed_dim=self.word_embed_dim,
                                      word_embed_trainable=False,
                                      use_crf=True, checkpoint_dir=self.checkpoint_dir)
        self.ner_model.build_model()
        self.ner_trainer = NERTrainer(self.ner_model, self.preprocessor)

        self.json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.json')
        self.weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.hdf5')

    def test_train(self):
        self.ner_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_crf(self):
        ner_model = BiLSTMCNNNER(self.num_class, self.char_embeddings, self.char_vocab_size,
                                 self.char_embed_dim, False, use_word=True,
                                 word_embeddings=self.word_embeddings,
                                 word_vocab_size=self.word_vocab_size,
                                 word_embed_dim=self.word_embed_dim,
                                 word_embed_trainable=False,
                                 use_crf=False, checkpoint_dir=self.checkpoint_dir)
        ner_model.build_model()
        ner_trainer = NERTrainer(ner_model, self.preprocessor)
        ner_trainer.train(self.train_data, self.train_labels, self.valid_data, self.valid_labels,
                          batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_word(self):
        preprocessor = NERPreprocessor(self.train_data+self.valid_data,
                                       self.valid_data+self.valid_labels, use_word=False,
                                       char_embed_type='word2vec')
        ner_model = BiLSTMCNNNER(self.num_class, self.char_embeddings, self.char_vocab_size,
                                 self.char_embed_dim, False, use_word=False,
                                 use_crf=True, checkpoint_dir=self.checkpoint_dir)
        ner_model.build_model()
        ner_trainer = NERTrainer(ner_model, preprocessor)
        ner_trainer.train(self.train_data, self.train_labels, self.valid_data, self.valid_labels,
                          batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_valid_data(self):
        self.ner_trainer.train(self.train_data, self.train_labels, batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_callbacks(self):
        self.ner_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7,
                               callbacks_str=['modelcheckpoint', 'earlystopping'])

        assert not os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        os.remove(self.weights_file)

    def test_train_swa(self):
        self.ner_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7, callbacks_str=['swa'])

        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

        json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner_swa.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner_swa.hdf5')
        assert not os.path.exists(json_file)
        assert os.path.exists(weights_file)
        self.ner_model.load_swa_model()
        os.remove(weights_file)

    def test_generator(self):
        self.ner_trainer.train_generator(self.train_data, self.train_labels,
                                         self.valid_data, self.valid_labels, batch_size=2, epochs=7,
                                         shuffle=True)

        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)
