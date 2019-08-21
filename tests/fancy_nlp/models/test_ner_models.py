# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.preprocessors import NERPreprocessor
from fancy_nlp.models.ner import BiLSTMNER, BiGRUCNNNER, BiLSTMCNNNER, BiGRUNER


class TestNerModel:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.test_file)
        self.preprocessor = NERPreprocessor(x_train, y_train, use_char=True, use_word=True,
                                            char_embed_type='word2vec', word_embed_type='word2vec')
        self.num_class = self.preprocessor.num_class
        self.char_embeddings = self.preprocessor.char_embeddings
        self.char_vocab_size = self.preprocessor.char_vocab_size
        self.char_embed_dim = self.preprocessor.char_embed_dim

        self.word_embeddings = self.preprocessor.word_embeddings
        self.word_vocab_size = self.preprocessor.word_vocab_size
        self.word_embed_dim = self.preprocessor.word_embed_dim
        self.checkpoint_dir = os.path.dirname(__file__)

    def test_bilstm_cnn_model(self):
        # no CRF, no word input
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 checkpoint_dir=self.checkpoint_dir,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=False,
                                 use_crf=False)
        ner_model.build_model()

        # CRF, no word input
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 checkpoint_dir=self.checkpoint_dir,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=False,
                                 use_crf=True)
        ner_model.build_model()

        # CRF, word
        ner_model = BiLSTMCNNNER(num_class=self.num_class,
                                 checkpoint_dir=self.checkpoint_dir,
                                 char_embeddings=self.char_embeddings,
                                 char_vocab_size=self.char_vocab_size,
                                 char_embed_dim=self.char_embed_dim,
                                 char_embed_trainable=False,
                                 use_word=True,
                                 word_embeddings=self.word_embeddings,
                                 word_vocab_size=self.word_vocab_size,
                                 word_embed_dim=self.word_embed_dim,
                                 word_embed_trainable=False,
                                 use_crf=True)
        ner_model.build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.hdf5')

        ner_model.save_model(json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        ner_model.load_model(json_file, weights_file)
        os.remove(json_file)
        os.remove(weights_file)

    def test_bigru_cnn_model(self):
        # no CRF, no word input
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                checkpoint_dir=self.checkpoint_dir,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_word=False,
                                use_crf=False)
        ner_model.build_model()

        # CRF, no word input
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                checkpoint_dir=self.checkpoint_dir,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_word=False,
                                use_crf=True)
        ner_model.build_model()

        # CRF, word
        ner_model = BiGRUCNNNER(num_class=self.num_class,
                                checkpoint_dir=self.checkpoint_dir,
                                char_embeddings=self.char_embeddings,
                                char_vocab_size=self.char_vocab_size,
                                char_embed_dim=self.char_embed_dim,
                                char_embed_trainable=False,
                                use_word=True,
                                word_embeddings=self.word_embeddings,
                                word_vocab_size=self.word_vocab_size,
                                word_embed_dim=self.word_embed_dim,
                                word_embed_trainable=False,
                                use_crf=True)
        ner_model.build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bigru_cnn_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bigru_cnn_ner.hdf5')

        ner_model.save_model(json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        ner_model.load_model(json_file, weights_file)
        os.remove(json_file)
        os.remove(weights_file)

    def test_bilstm_model(self):
        # no CRF, no word input
        ner_model = BiLSTMNER(num_class=self.num_class,
                              checkpoint_dir=self.checkpoint_dir,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_word=False,
                              use_crf=False)
        ner_model.build_model()

        # CRF, no word input
        ner_model = BiLSTMNER(num_class=self.num_class,
                              checkpoint_dir=self.checkpoint_dir,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_word=False,
                              use_crf=True)
        ner_model.build_model()

        # CRF, word
        ner_model = BiLSTMNER(num_class=self.num_class,
                              checkpoint_dir=self.checkpoint_dir,
                              char_embeddings=self.char_embeddings,
                              char_vocab_size=self.char_vocab_size,
                              char_embed_dim=self.char_embed_dim,
                              char_embed_trainable=False,
                              use_word=True,
                              word_embeddings=self.word_embeddings,
                              word_vocab_size=self.word_vocab_size,
                              word_embed_dim=self.word_embed_dim,
                              word_embed_trainable=False,
                              use_crf=True)
        ner_model.build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bilstm_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bilstm_ner.hdf5')

        ner_model.save_model(json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        ner_model.load_model(json_file, weights_file)
        os.remove(json_file)
        os.remove(weights_file)

    def test_bigru_model(self):
        # no CRF, no word input
        ner_model = BiGRUNER(num_class=self.num_class,
                             checkpoint_dir=self.checkpoint_dir,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_word=False,
                             use_crf=False)
        ner_model.build_model()

        # CRF, no word input
        ner_model = BiGRUNER(num_class=self.num_class,
                             checkpoint_dir=self.checkpoint_dir,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_word=False,
                             use_crf=True)
        ner_model.build_model()

        # CRF, word
        ner_model = BiGRUNER(num_class=self.num_class,
                             checkpoint_dir=self.checkpoint_dir,
                             char_embeddings=self.char_embeddings,
                             char_vocab_size=self.char_vocab_size,
                             char_embed_dim=self.char_embed_dim,
                             char_embed_trainable=False,
                             use_word=True,
                             word_embeddings=self.word_embeddings,
                             word_vocab_size=self.word_vocab_size,
                             word_embed_dim=self.word_embed_dim,
                             word_embed_trainable=False,
                             use_crf=True)
        ner_model.build_model()

        # save and load
        json_file = os.path.join(self.checkpoint_dir, 'bigru_ner.json')
        weights_file = os.path.join(self.checkpoint_dir, 'bigru_ner.hdf5')

        ner_model.save_model(json_file, weights_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        ner_model.load_model(json_file, weights_file)
        os.remove(json_file)
        os.remove(weights_file)
