# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.models.ner import BiLSTMCNNNER
from fancy_nlp.preprocessors import NERPreprocessor
from fancy_nlp.trainers import NERTrainer


class TestNERTrainer:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')
    bert_vocab_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/vocab.txt')
    bert_config_file = os.path.join(os.path.dirname(__file__),
                                    '../../../data/embeddings/bert_sample_model/bert_config.json')
    bert_model_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/bert_model.ckpt')

    def setup_class(self):
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = \
            load_ner_data_and_labels(self.test_file, split=True)
        self.preprocessor = NERPreprocessor(self.train_data+self.valid_data,
                                            self.train_labels+self.valid_labels,
                                            use_bert=True,
                                            use_word=True,
                                            bert_vocab_file=self.bert_vocab_file,
                                            char_embed_type='word2vec',
                                            word_embed_type='word2vec',
                                            max_len=16)
        self.num_class = self.preprocessor.num_class
        self.char_embeddings = self.preprocessor.char_embeddings
        self.char_vocab_size = self.preprocessor.char_vocab_size
        self.char_embed_dim = self.preprocessor.char_embed_dim

        self.word_embeddings = self.preprocessor.word_embeddings
        self.word_vocab_size = self.preprocessor.word_vocab_size
        self.word_embed_dim = self.preprocessor.word_embed_dim
        self.checkpoint_dir = os.path.dirname(__file__)

        self.ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                      use_char=True,
                                      char_embeddings=self.char_embeddings,
                                      char_vocab_size=self.char_vocab_size,
                                      char_embed_dim=self.char_embed_dim,
                                      char_embed_trainable=False,
                                      use_bert=True,
                                      bert_config_file=self.bert_config_file,
                                      bert_checkpoint_file=self.bert_model_file,
                                      use_word=True,
                                      word_embeddings=self.word_embeddings,
                                      word_vocab_size=self.word_vocab_size,
                                      word_embed_dim=self.word_embed_dim,
                                      word_embed_trainable=False,
                                      max_len=self.preprocessor.max_len,
                                      use_crf=True).build_model()

        self.swa_model = BiLSTMCNNNER(num_class=self.num_class,
                                      use_char=True,
                                      char_embeddings=self.char_embeddings,
                                      char_vocab_size=self.char_vocab_size,
                                      char_embed_dim=self.char_embed_dim,
                                      char_embed_trainable=False,
                                      use_bert=True,
                                      bert_config_file=self.bert_config_file,
                                      bert_checkpoint_file=self.bert_model_file,
                                      use_word=True,
                                      word_embeddings=self.word_embeddings,
                                      word_vocab_size=self.word_vocab_size,
                                      word_embed_dim=self.word_embed_dim,
                                      word_embed_trainable=False,
                                      max_len=self.preprocessor.max_len,
                                      use_crf=True).build_model()

        self.ner_trainer = NERTrainer(self.ner_model, self.preprocessor)

        self.json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.json')
        self.weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.hdf5')

    def test_train(self):
        self.ner_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_crf(self):
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 use_char=True,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_bert=True,
                                 bert_config_file=self.bert_config_file,
                                 bert_checkpoint_file=self.bert_model_file,
                                 use_word=True,
                                 word_embeddings=self.word_embeddings,
                                 word_vocab_size=self.word_vocab_size,
                                 word_embed_dim=self.word_embed_dim,
                                 word_embed_trainable=False,
                                 max_len=self.preprocessor.max_len,
                                 use_crf=False).build_model()

        ner_trainer = NERTrainer(ner_model, self.preprocessor)
        ner_trainer.train(self.train_data, self.train_labels, self.valid_data, self.valid_labels,
                          batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_word(self):
        preprocessor = NERPreprocessor(self.train_data+self.valid_data,
                                       self.train_labels+self.valid_labels,
                                       use_bert=True,
                                       use_word=False,
                                       bert_vocab_file=self.bert_vocab_file,
                                       max_len=16,
                                       char_embed_type='word2vec')
        ner_model = BiLSTMCNNNER(num_class=preprocessor.num_class,
                                 use_char=True,
                                 char_embeddings=preprocessor.char_embeddings,
                                 char_vocab_size=preprocessor.char_vocab_size,
                                 char_embed_dim=preprocessor.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_bert=True,
                                 bert_config_file=self.bert_config_file,
                                 bert_checkpoint_file=self.bert_model_file,
                                 use_word=False,
                                 word_embeddings=preprocessor.word_embeddings,
                                 word_vocab_size=preprocessor.word_vocab_size,
                                 word_embed_dim=preprocessor.word_embed_dim,
                                 word_embed_trainable=False,
                                 max_len=preprocessor.max_len,
                                 use_crf=True).build_model()

        ner_trainer = NERTrainer(ner_model, preprocessor)
        ner_trainer.train(self.train_data, self.train_labels, self.valid_data, self.valid_labels,
                          batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_bert(self):
        preprocessor = NERPreprocessor(self.train_data + self.valid_data,
                                       self.train_labels + self.valid_labels,
                                       use_word=True,
                                       char_embed_type='word2vec')
        ner_model = BiLSTMCNNNER(num_class=preprocessor.num_class,
                                 use_char=True,
                                 char_embeddings=preprocessor.char_embeddings,
                                 char_vocab_size=preprocessor.char_vocab_size,
                                 char_embed_dim=preprocessor.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=True,
                                 word_embeddings=preprocessor.word_embeddings,
                                 word_vocab_size=preprocessor.word_vocab_size,
                                 word_embed_dim=preprocessor.word_embed_dim,
                                 word_embed_trainable=False,
                                 max_len=preprocessor.max_len,
                                 use_crf=True).build_model()

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
                               callback_list=['modelcheckpoint', 'earlystopping'],
                               checkpoint_dir=os.path.dirname(__file__),
                               model_name='bilstm_cnn_ner')

        assert not os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        os.remove(self.weights_file)
        assert not os.path.exists(self.weights_file)

    def test_train_swa(self):
        self.ner_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7, callback_list=['swa'],
                               checkpoint_dir=os.path.dirname(__file__),
                               model_name='bilstm_cnn_ner',
                               swa_model=self.swa_model,
                               load_swa_model=True)

        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

        json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner_swa.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner_swa.hdf5')
        assert not os.path.exists(json_file)
        assert os.path.exists(weights_file)
        os.remove(weights_file)
        assert not os.path.exists(weights_file)

    def test_generator(self):
        self.ner_trainer.train_generator(self.train_data, self.train_labels,
                                         self.valid_data, self.valid_labels, batch_size=2, epochs=7)

        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)
