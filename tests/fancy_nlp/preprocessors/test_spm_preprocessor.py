# -*- coding: utf-8 -*-

import os
import jieba
import numpy as np
from fancy_nlp.utils.data_loader import load_spm_data_and_labels
from fancy_nlp.preprocessors.spm_preprocessor import SPMPreprocessor


class TestSPMPreprocessor:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/spm/webank/example.txt')
    bert_vocab_file = os.path.join(os.path.dirname(__file__),
                                   '../../../data/embeddings/bert_sample_model/vocab.txt')

    def setup_class(self):
        self.x_train, self.y_train = load_spm_data_and_labels(self.test_file)
        self.preprocessor = SPMPreprocessor(self.x_train, self.y_train, use_word=True,
                                            use_char=True, use_bert=False,
                                            bert_vocab_file=self.bert_vocab_file,
                                            external_word_dict=['微众'],
                                            word_embed_type='word2vec',
                                            max_len=16, max_word_len=3)

    def test_no_word(self):
        preprocessor = SPMPreprocessor(self.x_train, self.y_train, use_word=False,
                                       use_char=True, use_bert=True,
                                       bert_vocab_file=self.bert_vocab_file,
                                       external_word_dict=['微众'],
                                       char_embed_type='word2vec', max_len=16)

        assert len(preprocessor.char_vocab_count) + 4 == len(preprocessor.char_vocab) \
            == len(preprocessor.id2char)
        assert list(preprocessor.id2char.keys())[0] == 0
        for cnt in preprocessor.char_vocab_count.values():
            assert cnt >= 2
        assert preprocessor.char_embeddings.shape[0] == len(preprocessor.char_vocab)
        assert preprocessor.char_embeddings.shape[1] == 300
        assert not np.any(preprocessor.char_embeddings[0])
        assert preprocessor.word_embeddings is None

        assert len(preprocessor.label_vocab) == len(preprocessor.id2label)
        assert list(preprocessor.id2label.keys())[0] == 0

        features, y = preprocessor.prepare_input(preprocessor.train_data,
                                                      preprocessor.train_labels)
        assert len(features) == 6
        assert features[0].shape == features[1].shape == features[2].shape == features[3].shape == \
               features[4].shape == features[5].shape == \
               (len(self.x_train[0]), preprocessor.max_len)
        assert preprocessor.id2char[features[0][0][0]] == preprocessor.cls_token
        assert y.shape == (len(self.x_train[0]), preprocessor.num_class)

    def test_bert_model(self):
        preprocessor = SPMPreprocessor(self.x_train, self.y_train, use_word=False,
                                       use_char=False, use_bert=True, use_bert_model=True,
                                       bert_vocab_file=self.bert_vocab_file,
                                       max_len=16)

        assert preprocessor.word_embeddings is None
        assert preprocessor.char_embeddings is None

        assert len(preprocessor.label_vocab) == len(preprocessor.id2label)
        assert list(preprocessor.id2label.keys())[0] == 0

        features, y = preprocessor.prepare_input(self.x_train, self.y_train)
        assert len(features) == 2
        assert features[0].shape == features[1].shape == \
               (len(self.x_train[0]), preprocessor.max_len)
        assert y.shape == (len(self.x_train[0]), preprocessor.num_class)

    def test_no_bert(self):
        preprocessor = SPMPreprocessor(self.x_train, self.y_train, use_word=True,
                                       use_char=True, use_bert=False,
                                       bert_vocab_file=self.bert_vocab_file,
                                       external_word_dict=['微众'],
                                       word_embed_type='word2vec',
                                       max_len=16, max_word_len=3)

        assert len(preprocessor.word_vocab_count) + 2 == len(preprocessor.word_vocab) \
            == len(preprocessor.id2word)
        assert list(preprocessor.id2word.keys())[0] == 0
        for cnt in preprocessor.word_vocab_count.values():
            assert cnt >= 2
        assert preprocessor.word_embeddings.shape[0] == len(preprocessor.word_vocab)
        assert preprocessor.word_embeddings.shape[1] == 300
        assert not np.any(preprocessor.word_embeddings[0])

        assert len(preprocessor.char_vocab_count) + 2 == len(preprocessor.char_vocab) \
            == len(preprocessor.id2char)
        assert list(preprocessor.id2char.keys())[0] == 0
        for cnt in preprocessor.char_vocab_count.values():
            assert cnt >= 2
        assert preprocessor.char_embeddings is None

        assert len(preprocessor.label_vocab) == len(preprocessor.id2label)
        assert list(preprocessor.id2label.keys())[0] == 0

        features, y = preprocessor.prepare_input(self.x_train, self.y_train)
        assert len(features) == 4
        assert features[0].shape == features[2].shape == \
               (len(self.x_train[0]), preprocessor.max_len) and \
               features[1].shape == features[3].shape == \
               (len(self.x_train[0]), preprocessor.max_len, preprocessor.max_word_len)
        assert y.shape == (len(self.x_train[0]), preprocessor.num_class)

    def test_get_word_ids(self):
        example_text = ''.join(self.x_train[0][0])
        word_cut = jieba.lcut(example_text)
        word_ids = self.preprocessor.get_word_ids(word_cut)
        assert len(word_ids) == len(word_cut)

    def test_label_decode(self):
        rand_pred_probs = np.random.rand(2, self.preprocessor.num_class)
        pred_labels = self.preprocessor.label_decode(rand_pred_probs)
        assert isinstance(pred_labels[0], str)
        assert len(pred_labels) == 2

    def test_save_load(self):
        pkl_file = 'test_preprocessor.pkl'
        self.preprocessor.save(pkl_file)
        assert os.path.exists(pkl_file)
        new_preprocessor = SPMPreprocessor.load(pkl_file)
        assert new_preprocessor.num_class == self.preprocessor.num_class
        os.remove(pkl_file)
