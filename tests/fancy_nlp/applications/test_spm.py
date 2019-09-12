# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_spm_data_and_labels
from fancy_nlp.applications import SPM


class TestNER:
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

        self.checkpoint_dir = os.path.dirname(__file__)
        self.model_name = 'siamese_cnn_spm'
        self.json_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm.json')
        self.weights_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm.hdf5')
        self.swa_weights_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_spm_swa.hdf5')
        self.preprocessor_file = os.path.join(self.checkpoint_dir, 'siamese_cnn_preprocessor.pkl')

    def test_spm(self):
        spm = SPM()

        # test train word and char
        spm.fit(train_data=self.train_data,
                train_labels=self.train_labels,
                valid_data=self.valid_data,
                valid_labels=self.valid_labels,
                spm_model_type='siamese_cnn',
                use_word=True,
                word_embed_type='fasttext',
                use_char=True,
                use_bert=False,
                bert_vocab_file=self.bert_vocab_file,
                bert_config_file=self.bert_config_file,
                bert_checkpoint_file=self.bert_model_file,
                # max_len=16,
                # max_word_len=5,
                batch_size=2,
                epochs=10,
                callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
                checkpoint_dir=self.checkpoint_dir,
                model_name=self.model_name,
                load_swa_model=True)
        print(spm.preprocessor.max_len, spm.preprocessor.max_word_len)

        # test train char and bert
        spm.fit(train_data=self.train_data,
                train_labels=self.train_labels,
                valid_data=self.valid_data,
                valid_labels=self.valid_labels,
                spm_model_type='siamese_cnn',
                use_word=False,
                use_char=True,
                use_bert=True,
                bert_vocab_file=self.bert_vocab_file,
                bert_config_file=self.bert_config_file,
                bert_checkpoint_file=self.bert_model_file,
                max_len=16,
                max_word_len=5,
                batch_size=2,
                epochs=10,
                callback_list=['modelcheckpoint', 'earlystopping'],
                checkpoint_dir=self.checkpoint_dir,
                model_name=self.model_name,
                load_swa_model=True)

        # test train bert model
        spm.fit(train_data=self.train_data,
                train_labels=self.train_labels,
                valid_data=self.valid_data,
                valid_labels=self.valid_labels,
                spm_model_type='bert',
                use_word=False,
                use_char=False,
                use_bert=True,
                bert_vocab_file=self.bert_vocab_file,
                bert_config_file=self.bert_config_file,
                bert_checkpoint_file=self.bert_model_file,
                bert_output_layer_num=4,
                max_len=16,
                max_word_len=5,
                batch_size=2,
                epochs=10,
                callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
                checkpoint_dir=self.checkpoint_dir,
                model_name=self.model_name,
                load_swa_model=True)

        assert not os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        assert os.path.exists(self.swa_weights_file)
        os.remove(self.weights_file)
        os.remove(self.swa_weights_file)
        assert not os.path.exists(self.weights_file)
        assert not os.path.exists(self.swa_weights_file)

        # test score
        score = spm.score(self.valid_data, self.valid_labels)
        assert isinstance(score, (float, int))

        # test predict
        valid_label = spm.predict([self.valid_data[0][0], self.valid_data[1][0]])
        assert isinstance(valid_label, str)
        print([self.valid_data[0][0], self.valid_data[1][0]])
        print(valid_label)

        # test predict_batch
        valid_labels = spm.predict_batch(self.valid_data)
        assert isinstance(valid_labels, list) and isinstance(valid_labels[-1], str)
        assert len(valid_labels) == len(self.valid_data[0])
        assert valid_label == valid_labels[0]

        # test analyze
        valid_label = spm.analyze([self.valid_data[0][0], self.valid_data[1][0]])
        assert isinstance(valid_label, tuple)
        assert len(valid_label) == spm.preprocessor.num_class
        print(valid_label)
        print(self.valid_labels[0])

        # test analyze_batch
        valid_labels = spm.analyze_batch(self.valid_data)
        assert isinstance(valid_labels, list) and isinstance(valid_labels[-1], tuple)
        assert len(valid_labels) == len(self.valid_data[0])

        # test save
        spm.save(self.preprocessor_file, self.json_file, self.weights_file)
        assert os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        assert os.path.exists(self.preprocessor_file)

        # test load
        spm.load(self.preprocessor_file, self.json_file, self.weights_file)
        os.remove(self.json_file)
        os.remove(self.weights_file)
        os.remove(self.preprocessor_file)
