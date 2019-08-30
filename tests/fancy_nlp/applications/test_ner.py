# -*- coding: utf-8 -*-

import os
from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.applications import NER
from fancy_nlp.config import CACHE_DIR


class TestNER:
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

        self.checkpoint_dir = os.path.dirname(__file__)
        self.model_name = 'bilstm_cnn_ner'
        self.json_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.json')
        self.weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner.hdf5')
        self.swa_weights_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_ner_swa.hdf5')
        self.preprocessor_file = os.path.join(self.checkpoint_dir, 'bilstm_cnn_preprocessor.pkl')

    def test_ner(self):
        ner = NER()

        cache_dir = os.path.expanduser(CACHE_DIR)
        cache_subdir = 'pretrained_models'
        preprocessor_file = os.path.join(cache_dir, cache_subdir,
                                         'msra_ner_bilstm_cnn_crf_preprocessor.pkl')
        json_file = os.path.join(cache_dir, cache_subdir, 'msra_ner_bilstm_cnn_crf.json')

        weights_file = os.path.join(cache_dir, cache_subdir, 'msra_ner_bilstm_cnn_crf.hdf5')
        assert os.path.exists(preprocessor_file)
        assert os.path.exists(json_file)
        assert os.path.exists(weights_file)

        ner.analyze('同济大学位于上海市杨浦区，成立于1907年')
        ner.restrict_analyze('同济大学位于上海市杨浦区，成立于1907年')

        # test train
        ner.fit(train_data=self.train_data,
                train_labels=self.train_labels,
                valid_data=self.valid_data,
                valid_labels=self.valid_labels,
                ner_model_type='bilstm_cnn',
                use_char=True,
                use_bert=True,
                bert_vocab_file=self.bert_vocab_file,
                bert_config_file=self.bert_config_file,
                bert_checkpoint_file=self.bert_model_file,
                use_word=True,
                max_len=16,
                batch_size=2,
                epochs=7,
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
        score = ner.score(self.valid_data, self.valid_labels)
        assert isinstance(score, (float, int))

        # test predict
        valid_tag = ner.predict(self.valid_data[0])
        assert isinstance(valid_tag, list) and isinstance(valid_tag[0], str)
        assert len(valid_tag) == len(self.valid_data[0]) or \
               len(valid_tag) == ner.preprocessor.max_len - 2

        # test predict_batch
        valid_tags = ner.predict_batch(self.valid_data)
        assert isinstance(valid_tags, list) and isinstance(valid_tags[-1], list)
        assert isinstance(valid_tags[-1][0], str)
        assert len(valid_tags) == len(self.valid_data)
        assert len(valid_tags[-1]) == len(self.valid_data[-1]) or \
               len(valid_tags[-1]) == ner.preprocessor.max_len - 2

        # test analyze
        result = ner.analyze(self.valid_data[0])
        assert isinstance(result, dict) and 'text' in result and 'entities' in result

        # test analyze_batch
        results = ner.analyze_batch(self.valid_data)
        assert isinstance(results, list) and len(results) == len(self.valid_data)
        assert isinstance(results[-1], dict)
        assert 'text' in results[-1] and 'entities' in results[-1]

        # test restrict analyze
        result = ner.restrict_analyze(self.valid_data[0], threshold=0.85)
        entity_types = [entity['type'] for entity in result['entities']]
        assert len(set(entity_types)) == len(entity_types)
        for score in [entity['score'] for entity in result['entities']]:
            assert score >= 0.85

        # test restrict analyze batch
        results = ner.restrict_analyze_batch(self.valid_data, threshold=0.85)
        entity_types = [entity['type'] for entity in results[-1]['entities']]
        assert len(set(entity_types)) == len(entity_types)
        for score in [entity['score'] for entity in results[-1]['entities']]:
            assert score >= 0.85

        # test save
        ner.save(self.preprocessor_file, self.weights_file, self.json_file)
        assert os.path.exists(self.json_file)
        assert os.path.exists(self.weights_file)
        assert os.path.exists(self.preprocessor_file)

        # test load
        ner.load(self.preprocessor_file, self.weights_file, self.json_file)
        os.remove(self.json_file)
        os.remove(self.weights_file)
        os.remove(self.preprocessor_file)
