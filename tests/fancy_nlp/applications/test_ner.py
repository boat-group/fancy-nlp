# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.applications import NER


class TestNER:
    test_file = os.path.join(os.path.dirname(__file__), '../../../data/ner/msra/example.txt')

    def setup_class(self):
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = \
            load_ner_data_and_labels(self.test_file, split=True)

        self.checkpoint_dir = os.path.dirname(__file__)
        self.json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.json')
        self.weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.hdf5')
        self.swa_weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner_swa.hdf5')
        self.preprocessor_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_preprocessor.pkl')

    def test_ner(self):
        ner = NER(checkpoint_dir=self.checkpoint_dir, ner_model_type='bilstm_cnn', use_word=True,
                  use_crf=True)

        # test train
        ner.fit(train_data=self.train_data, train_labels=self.train_labels,
                valid_data=self.valid_data, valid_labels=self.valid_labels,
                batch_size=2, epochs=10, callbacks=['modelcheckpoint', 'earlystopping', 'swa'],
                load_swa_model=True, shuffle=True)

        assert not os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        assert os.path.exists(self.swa_weights_file)
        os.remove(self.weights_file)
        os.remove(self.swa_weights_file)
        assert not os.path.exists(self.weights_file)
        assert not os.path.exists(self.swa_weights_file)

        # test score
        score = ner.score(self.valid_data, self.valid_labels)
        print(score)
        assert isinstance(score, (float, int))

        # test predict
        valid_tag = ner.predict(self.valid_data[0])
        assert isinstance(valid_tag, list) and isinstance(valid_tag[0], str)
        assert len(valid_tag) == len(self.valid_data[0])

        # test predict_batch
        valid_tags = ner.predict_batch(self.valid_data)
        assert isinstance(valid_tags, list) and isinstance(valid_tags[-1], list)
        assert isinstance(valid_tags[-1][0], str)
        assert len(valid_tags) == len(self.valid_data)
        assert len(valid_tags[-1]) == len(self.valid_data[-1])

        # test analyze
        result = ner.analyze(self.valid_data[0])
        assert isinstance(result, dict) and 'chars' in result and 'entities' in result

        # test analyze_batch
        results = ner.analyze_batch(self.valid_data)
        assert isinstance(results, list) and len(results) == len(self.valid_data)
        assert isinstance(results[-1], dict)
        assert 'chars' in results[-1] and 'entities' in results[-1]

        # test save
        ner.save(os.path.basename(self.preprocessor_file), os.path.basename(self.weights_file),
                 os.path.basename(self.json_file))
        assert os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        assert os.path.exists(self.preprocessor_file)

        # test load
        ner.load(os.path.basename(self.preprocessor_file), os.path.basename(self.weights_file),
                 os.path.basename(self.json_file))
        os.remove(self.json_file)
        os.remove(self.weights_file)
        os.remove(self.preprocessor_file)
