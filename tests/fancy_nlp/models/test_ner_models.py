# -*- coding: utf-8 -*-

import os

from keras.models import clone_model

from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.preprocessors import NERPreprocessor
from fancy_nlp.models.ner import *
from fancy_nlp.utils import load_keras_model, save_keras_model, get_custom_objects


class TestNerModel:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')
    bert_vocab_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/vocab.txt')
    bert_config_file = os.path.join(os.path.dirname(__file__),
                                    '../../../data/embeddings/bert_sample_model/bert_config.json')
    bert_model_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/bert_model.ckpt')

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.test_file)
        self.preprocessor = NERPreprocessor(x_train, y_train, use_char=True, use_bert=True,
                                            use_word=True, bert_vocab_file=self.bert_vocab_file,
                                            char_embed_type='word2vec', word_embed_type='word2vec',
                                            max_len=16)
        self.num_class = self.preprocessor.num_class
        self.char_embeddings = self.preprocessor.char_embeddings
        self.char_vocab_size = self.preprocessor.char_vocab_size
        self.char_embed_dim = self.preprocessor.char_embed_dim

        self.word_embeddings = self.preprocessor.word_embeddings
        self.word_vocab_size = self.preprocessor.word_vocab_size
        self.word_embed_dim = self.preprocessor.word_embed_dim
        self.checkpoint_dir = os.path.dirname(__file__)

    def test_bilstm_cnn_model(self):
        # char, no CRF, no word input
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=False,
                                 use_crf=False).build_model()

        # char, CRF, no word, no bert input
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=False,
                                 use_crf=True).build_model()

        # char, CRF, word, no bert input
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=True,
                                 word_embeddings=self.word_embeddings,
                                 word_vocab_size=self.word_vocab_size,
                                 word_embed_dim=self.word_embed_dim,
                                 word_embed_trainable=False,
                                 use_crf=True).build_model()

        # char, CRF, word, bert
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_bert=True,
                                 bert_config_file=self.bert_config_file,
                                 bert_checkpoint_file=self.bert_model_file,
                                 bert_trainable=True,
                                 use_word=True,
                                 word_embeddings=self.word_embeddings,
                                 word_vocab_size=self.word_vocab_size,
                                 word_embed_dim=self.word_embed_dim,
                                 word_embed_trainable=False,
                                 max_len=16,
                                 use_crf=True).build_model()

        # test save and load
        json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.hdf5')

        save_keras_model(ner_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_bigru_cnn_model(self):
        # char, no CRF, no word input
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_word=False,
                                use_crf=False).build_model()

        # char, CRF, no word, no bert input
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_word=False,
                                use_crf=True).build_model()

        # char, CRF, word, no bert input
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_word=True,
                                word_embeddings=self.word_embeddings,
                                word_vocab_size=self.word_vocab_size,
                                word_embed_dim=self.word_embed_dim,
                                word_embed_trainable=False,
                                use_crf=True).build_model()

        # char, CRF, word, bert
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_bert=True,
                                bert_config_file=self.bert_config_file,
                                bert_checkpoint_file=self.bert_model_file,
                                bert_trainable=True,
                                use_word=True,
                                word_embeddings=self.word_embeddings,
                                word_vocab_size=self.word_vocab_size,
                                word_embed_dim=self.word_embed_dim,
                                word_embed_trainable=False,
                                max_len=16,
                                use_crf=True).build_model()
        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bigru_cnn_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bigru_cnn_ner.hdf5')

        save_keras_model(ner_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_bilstm_model(self):
        # char, no CRF, no word input
        ner_model = BiLSTMNER(num_class=self.num_class,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_word=False,
                              use_crf=False).build_model()

        # char, CRF, no word, no bert input
        ner_model = BiLSTMNER(num_class=self.num_class,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_word=False,
                              use_crf=True).build_model()

        # char, CRF, word, no bert input
        ner_model = BiLSTMNER(num_class=self.num_class,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_word=True,
                              word_embeddings=self.word_embeddings,
                              word_vocab_size=self.word_vocab_size,
                              word_embed_dim=self.word_embed_dim,
                              word_embed_trainable=False,
                              use_crf=True).build_model()

        # char, CRF, word, bert
        ner_model = BiLSTMNER(num_class=self.num_class,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_bert=True,
                              bert_config_file=self.bert_config_file,
                              bert_checkpoint_file=self.bert_model_file,
                              bert_trainable=True,
                              use_word=True,
                              word_embeddings=self.word_embeddings,
                              word_vocab_size=self.word_vocab_size,
                              word_embed_dim=self.word_embed_dim,
                              word_embed_trainable=False,
                              max_len=16,
                              use_crf=True).build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bilstm_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bilstm_ner.hdf5')

        save_keras_model(ner_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_bigru_model(self):
        # char, no CRF, no word input
        ner_model = BiGRUNER(num_class=self.num_class,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_word=False,
                             use_crf=False).build_model()

        # char, CRF, no word, no bert input
        ner_model = BiGRUNER(num_class=self.num_class,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_word=False,
                             use_crf=True).build_model()

        # char, CRF, word, no bert input
        ner_model = BiGRUNER(num_class=self.num_class,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_word=True,
                             word_embeddings=self.word_embeddings,
                             word_vocab_size=self.word_vocab_size,
                             word_embed_dim=self.word_embed_dim,
                             word_embed_trainable=False,
                             use_crf=True).build_model()

        # char, CRF, word, bert
        ner_model = BiGRUNER(num_class=self.num_class,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_bert=True,
                             bert_config_file=self.bert_config_file,
                             bert_checkpoint_file=self.bert_model_file,
                             bert_trainable=True,
                             use_word=True,
                             word_embeddings=self.word_embeddings,
                             word_vocab_size=self.word_vocab_size,
                             word_embed_dim=self.word_embed_dim,
                             word_embed_trainable=False,
                             max_len=16,
                             use_crf=True).build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bigru_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bigru_ner.hdf5')

        save_keras_model(ner_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)

    def test_bert_model(self):

        ner_model = BertNER(num_class=self.num_class,
                            bert_config_file=self.bert_config_file,
                            bert_checkpoint_file=self.bert_model_file,
                            bert_trainable=True,
                            max_len=16,
                            use_crf=True).build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bert_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bert_ner.hdf5')

        save_keras_model(ner_model, json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        load_keras_model(json_file, weights_file, custom_objects=get_custom_objects())
        os.remove(json_file)
        os.remove(weights_file)
        assert not os.path.exists(json_file)
        assert not os.path.exists(weights_file)
