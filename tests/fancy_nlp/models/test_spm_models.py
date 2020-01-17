# -*- coding: utf-8 -*-

import os

from fancy_nlp.utils import load_spm_data_and_labels
from fancy_nlp.preprocessors import SPMPreprocessor
from fancy_nlp.models.spm import *
from fancy_nlp.utils import load_keras_model, save_keras_model, get_custom_objects


class TestSpmModel:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/spm/webank/example.txt')
    bert_vocab_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/vocab.txt')
    bert_config_file = os.path.join(os.path.dirname(__file__),
                                    '../../../data/embeddings/bert_sample_model/bert_config.json')
    bert_model_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/bert_model.ckpt')

    def setup_class(self):
        x_train, y_train = load_spm_data_and_labels(self.test_file)
        self.preprocessor = SPMPreprocessor(x_train, y_train, use_word=True, use_char=True, use_bert=False,
                                            bert_vocab_file=self.bert_vocab_file,
                                            char_embed_type='word2vec', word_embed_type='word2vec',
                                            max_len=10)
        self.num_class = self.preprocessor.num_class
        self.char_embeddings = self.preprocessor.char_embeddings
        self.char_vocab_size = self.preprocessor.char_vocab_size
        self.char_embed_dim = self.preprocessor.char_embed_dim

        self.word_embeddings = self.preprocessor.word_embeddings
        self.word_vocab_size = self.preprocessor.word_vocab_size
        self.word_embed_dim = self.preprocessor.word_embed_dim
        self.checkpoint_dir = os.path.dirname(__file__)

    def test_siamese_cnn_model(self):
        # word, char
        spm_model = SiameseCNN(num_class=self.num_class,
                               use_word=True,
                               word_embeddings=self.word_embeddings,
                               word_vocab_size=self.word_vocab_size,
                               word_embed_dim=self.word_embed_dim,
                               word_embed_trainable=False,
                               use_char=True,
                               char_embeddings=self.char_embeddings,
                               char_vocab_size=self.char_vocab_size,
                               char_embed_dim=self.char_embed_dim,
                               char_embed_trainable=False,
                               use_bert=False,
                               max_len=10).build_model()

        # char, bert
        spm_model = SiameseCNN(num_class=self.num_class,
                               use_word=False,
                               use_char=True,
                               char_embeddings=self.char_embeddings,
                               char_vocab_size=self.char_vocab_size,
                               char_embed_dim=self.char_embed_dim,
                               char_embed_trainable=False,
                               use_bert=True,
                               bert_config_file=self.bert_config_file,
                               bert_checkpoint_file=self.bert_model_file,
                               max_len=10).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm.json')
        weights_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm.hdf5')

        save_keras_model(spm_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_siamese_bilstm_model(self):
        # word, char
        spm_model = SiameseBiLSTM(num_class=self.num_class,
                                  use_word=True,
                                  word_embeddings=self.word_embeddings,
                                  word_vocab_size=self.word_vocab_size,
                                  word_embed_dim=self.word_embed_dim,
                                  word_embed_trainable=False,
                                  use_char=True,
                                  char_embeddings=self.char_embeddings,
                                  char_vocab_size=self.char_vocab_size,
                                  char_embed_dim=self.char_embed_dim,
                                  char_embed_trainable=False,
                                  use_bert=False,
                                  max_len=10).build_model()

        # char, bert
        spm_model = SiameseBiLSTM(num_class=self.num_class,
                                  use_word=False,
                                  use_char=True,
                                  char_embeddings=self.char_embeddings,
                                  char_vocab_size=self.char_vocab_size,
                                  char_embed_dim=self.char_embed_dim,
                                  char_embed_trainable=False,
                                  use_bert=True,
                                  bert_config_file=self.bert_config_file,
                                  bert_checkpoint_file=self.bert_model_file,
                                  max_len=10).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'siamese_bilstm_spm.json')
        weights_file = os.path.join(self.checkpoint_dir, 'siamese_bilstm_spm.hdf5')

        save_keras_model(spm_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_siamese_bigru_model(self):
        # word, char
        spm_model = SiameseBiGRU(num_class=self.num_class,
                                 use_word=True,
                                 word_embeddings=self.word_embeddings,
                                 word_vocab_size=self.word_vocab_size,
                                 word_embed_dim=self.word_embed_dim,
                                 word_embed_trainable=False,
                                 use_char=True,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_bert=False,
                                 max_len=10).build_model()

        # char, bert
        spm_model = SiameseBiGRU(num_class=self.num_class,
                                 use_word=False,
                                 use_char=True,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_bert=True,
                                 bert_config_file=self.bert_config_file,
                                 bert_checkpoint_file=self.bert_model_file,
                                 max_len=10).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'siamese_bigru_spm.json')
        weights_file = os.path.join(self.checkpoint_dir, 'siamese_bigru_spm.hdf5')

        save_keras_model(spm_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_esim_model(self):
        # word, char
        spm_model = ESIM(num_class=self.num_class,
                         use_word=True,
                         word_embeddings=self.word_embeddings,
                         word_vocab_size=self.word_vocab_size,
                         word_embed_dim=self.word_embed_dim,
                         word_embed_trainable=False,
                         use_char=True,
                         char_embeddings=self.char_embeddings,
                         char_vocab_size=self.char_vocab_size,
                         char_embed_dim=self.char_embed_dim,
                         char_embed_trainable=False,
                         use_bert=False,
                         max_len=10).build_model()

        # char, bert
        spm_model = ESIM(num_class=self.num_class,
                         use_word=False,
                         use_char=True,
                         char_embeddings=self.char_embeddings,
                         char_vocab_size=self.char_vocab_size,
                         char_embed_dim=self.char_embed_dim,
                         char_embed_trainable=False,
                         use_bert=True,
                         bert_config_file=self.bert_config_file,
                         bert_checkpoint_file=self.bert_model_file,
                         max_len=10).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'esim_spm.json')
        weights_file = os.path.join(self.checkpoint_dir, 'esim_spm.hdf5')

        save_keras_model(spm_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_bimpm_model(self):
        # word, char
        spm_model = BiMPM(num_class=self.num_class,
                          use_word=True,
                          word_embeddings=self.word_embeddings,
                          word_vocab_size=self.word_vocab_size,
                          word_embed_dim=self.word_embed_dim,
                          word_embed_trainable=False,
                          use_char=True,
                          char_embeddings=self.char_embeddings,
                          char_vocab_size=self.char_vocab_size,
                          char_embed_dim=self.char_embed_dim,
                          char_embed_trainable=False,
                          use_bert=False,
                          max_len=10).build_model()

        # char, bert
        spm_model = BiMPM(num_class=self.num_class,
                          use_word=False,
                          use_char=True,
                          char_embeddings=self.char_embeddings,
                          char_vocab_size=self.char_vocab_size,
                          char_embed_dim=self.char_embed_dim,
                          char_embed_trainable=False,
                          use_bert=True,
                          bert_config_file=self.bert_config_file,
                          bert_checkpoint_file=self.bert_model_file,
                          max_len=10).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'bimpm_spm.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bimpm_spm.hdf5')

        save_keras_model(spm_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_bert_model(self):
        spm_model = BertSPM(num_class=self.num_class,
                            bert_config_file=self.bert_config_file,
                            bert_checkpoint_file=self.bert_model_file,
                            bert_trainable=True,
                            max_len=10).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'bert_spm.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bert_spm.hdf5')

        save_keras_model(spm_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)
