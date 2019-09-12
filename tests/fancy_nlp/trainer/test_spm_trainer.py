# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_spm_data_and_labels
from fancy_nlp.models.spm import SiameseCNN, BertSPM
from fancy_nlp.preprocessors import SPMPreprocessor
from fancy_nlp.trainers import SPMTrainer


class TestSPMTrainer:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/spm/webank/example.txt')
    bert_vocab_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/vocab.txt')
    bert_config_file = os.path.join(os.path.dirname(__file__),
                                    '../../../data/embeddings/bert_sample_model/bert_config.json')
    bert_model_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/bert_model.ckpt')

    def setup_class(self):
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = \
            load_spm_data_and_labels(self.test_file, split_mode=1)
        self.preprocessor = SPMPreprocessor((self.train_data[0] + self.valid_data[0],
                                             self.train_data[1] + self.valid_data[1]),
                                            self.train_labels + self.valid_labels,
                                            use_word=True,
                                            use_char=True,
                                            bert_vocab_file=self.bert_vocab_file,
                                            word_embed_type='word2vec',
                                            char_embed_type='word2vec',
                                            max_len=16)
        self.num_class = self.preprocessor.num_class
        self.char_embeddings = self.preprocessor.char_embeddings
        self.char_vocab_size = self.preprocessor.char_vocab_size
        self.char_embed_dim = self.preprocessor.char_embed_dim

        self.word_embeddings = self.preprocessor.word_embeddings
        self.word_vocab_size = self.preprocessor.word_vocab_size
        self.word_embed_dim = self.preprocessor.word_embed_dim
        self.checkpoint_dir = os.path.dirname(__file__)

        self.spm_model = SiameseCNN(num_class=self.num_class,
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
                                    bert_config_file=self.bert_config_file,
                                    bert_checkpoint_file=self.bert_model_file,
                                    bert_trainable=True,
                                    max_len=self.preprocessor.max_len,
                                    max_word_len=self.preprocessor.max_word_len).build_model()

        self.swa_model = SiameseCNN(num_class=self.num_class,
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
                                    bert_config_file=self.bert_config_file,
                                    bert_checkpoint_file=self.bert_model_file,
                                    bert_trainable=True,
                                    max_len=self.preprocessor.max_len,
                                    max_word_len=self.preprocessor.max_word_len).build_model()

        self.spm_trainer = SPMTrainer(self.spm_model, self.preprocessor)

        self.json_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm.json')
        self.weights_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm.hdf5')

    def test_train(self):
        self.spm_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_word(self):
        preprocessor = SPMPreprocessor((self.train_data[0] + self.valid_data[0],
                                        self.train_data[1] + self.valid_data[1]),
                                       self.train_labels + self.valid_labels,
                                       use_word=False,
                                       use_char=True,
                                       use_bert=True,
                                       bert_vocab_file=self.bert_vocab_file,
                                       char_embed_type='word2vec',
                                       max_len=16)
        self.num_class = preprocessor.num_class
        self.char_embeddings = preprocessor.char_embeddings
        self.char_vocab_size = preprocessor.char_vocab_size
        self.char_embed_dim = preprocessor.char_embed_dim

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
                               bert_trainable=True,
                               max_len=preprocessor.max_len).build_model()

        spm_trainer = SPMTrainer(spm_model, preprocessor)
        spm_trainer.train(self.train_data, self.train_labels, self.valid_data, self.valid_labels,
                          batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_bert_model(self):
        preprocessor = SPMPreprocessor((self.train_data[0] + self.valid_data[0],
                                        self.train_data[1] + self.valid_data[1]),
                                       self.train_labels + self.valid_labels,
                                       use_word=False,
                                       use_char=False,
                                       use_bert=True,
                                       use_bert_model=True,
                                       bert_vocab_file=self.bert_vocab_file,
                                       max_len=16)
        spm_model = BertSPM(num_class=self.num_class,
                            bert_config_file=self.bert_config_file,
                            bert_checkpoint_file=self.bert_model_file,
                            bert_trainable=True,
                            max_len=preprocessor.max_len).build_model()

        spm_trainer = SPMTrainer(spm_model, preprocessor)
        spm_trainer.train(self.train_data, self.train_labels, self.valid_data, self.valid_labels,
                          batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_no_valid_data(self):
        self.spm_trainer.train(self.train_data, self.train_labels, batch_size=2, epochs=7)
        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

    def test_train_callbacks(self):
        self.spm_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7,
                               callback_list=['modelcheckpoint', 'earlystopping'],
                               checkpoint_dir=os.path.dirname(__file__),
                               model_name='siamese_cnn_spm')

        assert not os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        os.remove(self.weights_file)
        assert not os.path.exists(self.weights_file)

    def test_train_swa(self):
        self.spm_trainer.train(self.train_data, self.train_labels, self.valid_data,
                               self.valid_labels, batch_size=2, epochs=7, callback_list=['swa'],
                               checkpoint_dir=os.path.dirname(__file__),
                               model_name='siamese_cnn_spm',
                               swa_model=self.swa_model,
                               load_swa_model=True)

        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)

        json_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm_swa.json')
        weights_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm_swa.hdf5')
        assert not os.path.exists(json_file)
        assert os.path.exists(weights_file)
        os.remove(weights_file)
        assert not os.path.exists(weights_file)

    def test_generator(self):
        self.spm_trainer.train_generator(self.train_data, self.train_labels,
                                         self.valid_data, self.valid_labels, batch_size=2, epochs=7)

        assert not os.path.exists(self.json_file)
        assert not os.path.exists(self.weights_file)
