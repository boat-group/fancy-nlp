# -*- coding: utf-8 -*-

from typing import List, Optional, Union, Dict, Any

from absl import logging
import numpy as np
import tensorflow as tf

from fancy_nlp.preprocessors import NERPreprocessor
from fancy_nlp.models.ner import ner_model_dict
from fancy_nlp.trainers import NERTrainer
from fancy_nlp.predictors import NERPredictor
from fancy_nlp.utils import get_custom_objects
from fancy_nlp.config import CACHE_DIR


class NER(object):
    """NER application. Support training ner model from scratch with provided dataset,
    loading pre-trained ner model as well as evaluating ner model on raw text with detailed and
    pretty-formatted tagging results.

    Examples:

    The following snippet shows how to train a ner model that uses BiLSTM-CNN-CRF model with
    character embedding and bert embedding as input and save it to disk:

    ```python
        from fancy_nlp.utils import load_ner_data_and_labels
        from fancy_nlp.applications import NER

        msra_train_file = 'data/ner/msra/train_data'
        msra_dev_file = 'data/ner/msra/test_data'
        bert_vocab_file='data/embeddings/chinese_L-12_H-768_A-12/vocab.txt',
        bert_config_file='data/embeddings/chinese_L-12_H-768_A-12/bert_config.json'),
        bert_checkpoint_file='data/embeddings/chinese_L-12_H-768_A-12/bert_model.ckpt')

        checkpoint_dir = 'ner_models'
        model_name = 'bert-bilstm-cnn-crf'
        weights_file = os.path.join(checkpoint_dir, f'{model_name}.hdf5')
        json_file = os.path.join(checkpoint_dir, f'{model_name}.json')
        preprocessor_file = os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl')

        # start training
        train_data, train_labels = load_ner_data_and_labels(msra_train_file, delimiter='\t')
        dev_data, dev_labels = load_ner_data_and_labels(msra_dev_file, delimiter='\t')
        ner = NER()
        ner.fit(train_data=train_data,
                train_labels=train_labels,
                dev_data=dev_data,
                dev_labels=dev_labels,
                ner_model_type='bilstm_cnn',
                use_char=True,
                use_bert=True,
                bert_vocab_file=bert_vocab_file,
                bert_config_file=bert_config_file,
                bert_checkpoint_file=bert_checkpoint_file,
                bert_trainable=True,
                use_crf=True,
                callback_list=['modelcheckpoint', 'earlystopping'],
                checkpoint_dir=checkpoint_dir,
                model_name=model_name)

        # save ner application's preprocessor, model architecture and model weights to disk
        # with `ModelCheckpoint` callback, model weights will be saved to disk after training.
        # In that case, we don't need to save it again. So we pass None to weight_file
        ner.save(preprocessor_file=preprocessor_file, json_file=json_file, weight_file=None)
    ```

    The following snippet shows how to load a pre-trained ner model from disk and evaluate it using
    raw text:

    ```python
        from fancy_nlp.utils import load_ner_data_and_labels
        from fancy_nlp.applications import NER

        checkpoint_dir = 'ner_models'
        model_name = 'bert-bilstm-cnn-crf'
        weights_file = os.path.join(checkpoint_dir, f'{model_name}.hdf5')
        json_file = os.path.join(checkpoint_dir, f'{model_name}.json')
        preprocessor_file = os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl')

        ner = NER()
        # load from disk
        ner.load(preprocessor_file=preprocessor_file, json_file=json_file, weight_file=weight_file)

        # evaluate over development dataset
        msra_dev_file = 'data/ner/msra/test_data'
        dev_data, dev_labels = load_ner_data_and_labels(msra_dev_file, delimiter='\t')
        print(ner.score(valid_data=dev_data, valid_labels=dev_labels))

        # predict tad sequence for given text
        print(ner.predict(text='同济大学位于上海市杨浦区，校长为陈杰')

        # show detailed tagging result in pretty-formatted for given text
        print(ner.analyze(text='同济大学位于上海市杨浦区，校长为陈杰'))
    ```

    """

    def __init__(self, use_pretrained: bool = False) -> None:
        """

        Args:
            use_pretrained: Boolean. Whether to load a pre-trained ner model that was trained on
                msra dataset using BiLSTM+CNN+CRF model.
        """
        # instance of NERPreprocessor, used to process the dataset and prepare model input
        self.preprocessor = None
        # instance of tf.Keras Model, ner model, the core of ner application
        self.model = None
        # instance of NERTrainer, used to train the ner model with dataset
        self.trainer = None
        # instance of NERPredictor, used to predict tagging results with the trained ner model
        self.predictor = None

        if use_pretrained:
            self.load_pretrained_model()

    def fit(self,
            train_data: List[List[str]],
            train_labels: List[List[str]],
            valid_data: Optional[List[List[str]]] = None,
            valid_labels: Optional[List[List[str]]] = None,
            ner_model_type: str = 'bilstm_cnn',
            use_char: bool = True,
            char_embed_type: Optional[str] = 'word2vec',
            char_embed_dim: int = 300,
            char_embed_trainable: bool = True,
            use_bert: bool = False,
            bert_vocab_file: Optional[str] = None,
            bert_config_file: Optional[str] = None,
            bert_checkpoint_file: Optional[str] = None,
            bert_trainable: bool = False,
            use_word: bool = False,
            external_word_dict: Optional[List[str]] = None,
            word_embed_type: Optional[str] = 'word2vec',
            word_embed_dim: int = 300,
            word_embed_trainable: bool = True,
            max_len: Optional[int] = None,
            use_crf: bool = True,
            optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
            batch_size: int = 32,
            epochs: int = 50,
            callback_list: Optional[List[str]] = None,
            checkpoint_dir: Optional[str] = None,
            model_name: Optional[str] = None,
            load_swa_model: bool = False,
            **kwargs) -> None:
        """Train ner model with provided dataset.

        We would like to make NER in Fancy-NLP more configurable, so we provided a bunch of
        arguments for users to configure:
            1. Which type of ner model to use; Currently we implement 5 types of ner models:
            'bilstm', 'bisltm-cnn', 'bigru', 'bigru-cnn' and 'bert'.

            2. Which kind of input embedding to use; We support 3 kinds of embedding: char
            embedding, bert embedding and word embedding. We can choose any one of them or
            combine any two or all of them to used as input. Note that our ner model only support
            char-level input, for the reason that char-level input haven shown
            effective for Chinese NER task without word-segmentation error. Therefore, we should use
            char embedding or bert embedding or both of them as main input. On that basis, we
            can use word embedding as auxiliary input to provide semantic information.

            3. Whether to use CRF; We can choose whether to add crf layer on the last layer of ner
            model.

            4. How to train the model:
            a) which optimizer to use, we support any optimizer that is compatible with
               tf.keras's optimizer:
               https://www.tensorflow.org/api_docs/python/tf/keras/optimizers ;
            b) how many sample to train per batch, how many epoch to train;
            c) which callbacks to use during training, we currently support 3 kinds of callbacks:
               i) 'modelcheckpoint' is used to save the model with best performance:
                   https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint;
               ii) 'earlystoppoing' is used to stop training when no performance gain observed：
                   https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
               iii) 'swa' is used to apply an novel weight averaging ensemble mechanism to the
                   ner model we are training: https://arxiv.org/abs/1803.05407)

            5. Where to save the model

        Args:
            train_data: List of List of str. List of tokenized (in char level) texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            train_labels: List of List of str. The labels of train_data, usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
            valid_data: Optional List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation.
            valid_labels: Optional List of List of str, can be None. The labels of valid_data.
                We can use fancy_nlp.utils.load_ner_data_and_labels() function to get training
                or validation data and labels from raw dataset in CoNLL format.
            ner_model_type: str. Which type of ner model to use, can be one of {'bilstm',
                'bilstm-cnn', 'bigru', 'bigru-cnn', 'bert'}.
            use_char: Boolean. Whether to use character embedding as input.
            char_embed_type: Optional str, can be None. The type of char embedding, can be a
                pre-trained embedding filename that used to load pre-trained embedding,
                or a embedding training method (one of {'word2vec', 'fasttext'}) that used to
                train character embedding with dataset. If None, do not apply anr pre-trained
                embedding, and use randomly initialized embedding instead.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_vocab_file: Optional str, can be None. Path to bert's vocabulary file.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            external_word_dict: Optional List of str, can be None. List of words, external word
                dictionary that will be used to loaded in jieba. It can be regarded as one kind
                of gazetter that contain a number of correct named-entities.
                Such as ``['南京市', '长江大桥']``
            word_embed_dim: similar as 'char_embed_dim'.
            word_embed_type: similar as 'char_embed_type'.
            word_embed_trainable: similar as 'char_embed_trainable'.
            max_len: Optional int, can be None. Max length of one sequence. If None, we dynamically
                use the max length of each batch as max_len. However, max_len must be provided
                when using bert as input.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model
            callback_list: Optional List of str or instance of `keras.callbacks.Callback`,
                can be None. Each item indicates the callback to apply during training. Currently,
                we support using 'modelcheckpoint' for `ModelCheckpoint` callback, 'earlystopping`
                for 'Earlystopping` callback, 'swa' for 'SWA' callback. We will automatically add
                `NERMetric` callback when valid_data and valid_labels are both provided.
            checkpoint_dir: Optional str, can be None. Directory to save the ner model. It must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training.
            model_name: Optional str, can be None. Prefix of ner model's weights file. I must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training. For example, if checkpoint_dir is 'ckpt' and
                model_name is 'model', the weights of ner model saved by `ModelCheckpoint` callback
                will be 'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'.
            load_swa_model: Boolean. Whether to load swa model, only apply when using `SWA`
                Callback. We suggest set it to True when using `SWA` Callback since swa model
                performs better than the original model at most cases.
            **kwargs: Other argument for building ner model, such as "rnn_units", "fc_dim" etc. See
                models.ner_models.py for there arguments.
        """

        # whether to use traditional bert model for prediction
        use_bert_model = ner_model_type == 'bert'
        # add assertion for checking input
        assert not (use_bert_model and use_word), \
            'when using bert model, `use_word` must be False'
        assert not (use_bert_model and use_char), \
            'when using bert model, `use_char` must be False'
        assert not (use_bert_model and not use_bert), \
            'when using bert model, `use_bert` must be True'

        self.preprocessor = NERPreprocessor(train_data=train_data,
                                            train_labels=train_labels,
                                            use_char=use_char,
                                            use_bert=use_bert,
                                            use_word=use_word,
                                            external_word_dict=external_word_dict,
                                            bert_vocab_file=bert_vocab_file,
                                            char_embed_type=char_embed_type,
                                            char_embed_dim=char_embed_dim,
                                            word_embed_type=word_embed_type,
                                            word_embed_dim=word_embed_dim,
                                            max_len=max_len)

        self.model = self.get_ner_model(ner_model_type=ner_model_type,
                                        num_class=self.preprocessor.num_class,
                                        use_char=use_char,
                                        char_embeddings=self.preprocessor.char_embeddings,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embed_dim=self.preprocessor.char_embed_dim,
                                        char_embed_trainable=char_embed_trainable,
                                        use_bert=use_bert,
                                        bert_config_file=bert_config_file,
                                        bert_checkpoint_file=bert_checkpoint_file,
                                        bert_trainable=bert_trainable,
                                        use_word=use_word,
                                        word_embeddings=self.preprocessor.word_embeddings,
                                        word_vocab_size=self.preprocessor.word_vocab_size,
                                        word_embed_dim=self.preprocessor.word_embed_dim,
                                        word_embed_trainable=word_embed_trainable,
                                        max_len=self.preprocessor.max_len,
                                        use_crf=use_crf,
                                        optimizer=optimizer,
                                        **kwargs)

        if 'swa' in callback_list:
            # initialize swa model when using `SWA` callback
            swa_model = self.get_ner_model(ner_model_type=ner_model_type,
                                           num_class=self.preprocessor.num_class,
                                           use_char=use_char,
                                           char_embeddings=self.preprocessor.char_embeddings,
                                           char_vocab_size=self.preprocessor.char_vocab_size,
                                           char_embed_dim=self.preprocessor.char_embed_dim,
                                           char_embed_trainable=char_embed_trainable,
                                           use_bert=use_bert,
                                           bert_config_file=bert_config_file,
                                           bert_checkpoint_file=bert_checkpoint_file,
                                           bert_trainable=bert_trainable,
                                           use_word=use_word,
                                           word_embeddings=self.preprocessor.word_embeddings,
                                           word_vocab_size=self.preprocessor.word_vocab_size,
                                           word_embed_dim=self.preprocessor.word_embed_dim,
                                           word_embed_trainable=word_embed_trainable,
                                           max_len=self.preprocessor.max_len,
                                           use_crf=use_crf,
                                           optimizer=optimizer,
                                           **kwargs)
        else:
            swa_model = None

        self.trainer = NERTrainer(self.model, self.preprocessor)
        self.trainer.train_generator(train_data, train_labels, valid_data, valid_labels,
                                     batch_size, epochs, callback_list, checkpoint_dir, model_name,
                                     swa_model, load_swa_model)

        self.predictor = NERPredictor(self.model, self.preprocessor)

        if valid_data is not None and valid_labels is not None:
            logging.info('Evaluating on validation data...')
            self.score(valid_data, valid_labels)

    def score(self, data: List[List[str]], labels: List[List[str]]) -> float:
        """Evaluate the performance of ner model with given data and labels, return the f1 score.

        Args:
            data: List of List of str. List of tokenized (in char level) texts ,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.
            labels: List of List of str. The corresponding labels , usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.

        Returns:
            Float. The F1 score.

        """
        if self.trainer:
            return self.trainer.evaluate(data, labels)
        else:
            logging.fatal('Trainer is None! Call fit() or load() to get trainer.')

    def predict(self, text: Union[str, List[str]]) -> List[str]:
        """Return the tag sequence of given text predicted by the ner model

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``.

        Returns:
            List of str. The tag sequence, like ``['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']``

        """
        if self.predictor:
            return self.predictor.tag(text)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def predict_batch(self, texts: Union[List[str], List[List[str]]]) -> List[List[str]]:
        """Return the tag sequences of given batch of texts predicted by the ner model

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.

        Returns:
            List of List of str. The tag sequences, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O',
            'O']]``

        """
        if self.predictor:
            return self.predictor.tag_batch(texts)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def analyze(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Analyze the tagging result of given text predicted by the ner model and return the
        result in pretty format with detailed information.

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``.

        Returns:
            A Dict including the original text and list of recognized entities with detailed
            information (name, type, score, offset). Specifically, it will be like:
            {'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
            }

        Notes:
            the score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution.

        """
        if self.predictor:
            return self.predictor.pretty_tag(text)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def analyze_batch(self, texts: Union[List[str], List[List[str]]]) -> List[Dict[str, Any]]:
        """Analyze the tagging results of given batch of text predicted by the ner model and
        return the results in pretty format with detailed information.

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``.

        Returns:
            List of Dict. Each Dict contain the tagging results of one text, including the original
            text and list of recognized entities with detailed information (name, type, score,
            offset). Specifically, it will be like:
            [{'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
             }
             ...
            ]

        Notes:
            The score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution.

        """
        if self.predictor:
            return self.predictor.pretty_tag_batch(texts)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def restrict_analyze(self,
                         text: Union[str, List[str]],
                         threshold: float = 0.85) -> Dict[str, Any]:
        """Analyze the tagging result of given text predicted by the ner model and then remove some
        recognized entities such that 1) all entities's scores are higher than threshold; 2)
        for each entity type, only keep the entity with the highest score. After that, return the
        recognized result in pretty format with detailed information.

        Args:
            text: str or List of str. Can be a un-tokenized text, like ``'我在上海上学'`` or a
                tokenized (in char level) text sequence, like ``['我', '在', '上', '海', '上', '学']``.
            threshold: float. The scores of recognized entities must be higher than threshold.

        Returns:
            A Dict including the original text and list of recognized entities with detailed
            information (name, type, score, offset). Specifically, it will be like:
            {'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
            }

        Notes:
            The score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution. As a result,
            we do not recommend you use this function when using crf layer.

        """
        if self.predictor:
            return self.predictor.restrict_tag(text, threshold)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def restrict_analyze_batch(self,
                               texts: Union[List[str], List[List[str]]],
                               threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Analyze the tagging results of given batch of texts predicted by the ner model and then
        remove some recognized entities such that 1) all entities's scores are higher than
        threshold; 2) for each entity type, only keep the entity with the highest score. After
        that, return the recognized results in pretty format with detailed information.

        Args:
            texts: List of str or List of List of str. Can be a batch of un-tokenized texts,
                like ``['我在上海上学', ...]`` or a batch of tokenized (in char level) text sequences,
                like ``[['我', '在', '上', '海', '上', '学'], ...]``
            threshold: float. The scores of recognized entities must be higher than threshold.

        Returns:
            List of Dict. Each Dict contain the tagging results of one text, including the original
            text and list of recognized entities with detailed information (name, type, score,
            offset). Specifically, it will be like:
            [{'text': '我在上海上学',
             'entities': [{'name': '上海',
                           'type': 'LOC',
                           'score': 0.9986118674278259,
                           'beginOffset': 2,
                           'endOffset': 4
                           }]
             }
             ...
            ]

        Notes:
            The score of entity is the probability of being a named-entity, it is computed by
            taking the average the probability of all the tokens within the entity, which is
            predicted by the ner model. However, if one use crf layer at the last layer of ner
            model, the score will be always 1. This is because the viterbi algorithm used by crf
            will output a definite best path instead of probability distribution. As a result,
            we do not recommend you use this function when using crf layer.

        """
        if self.predictor:
            return self.predictor.restrict_tag_batch(texts, threshold)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def save(self,
             preprocessor_file: str,
             json_file: str,
             weights_file: Optional[str] = None) -> None:
        """Save ner application to disk.

        There are 3 things in total that we need to save: 1) preprocessor, which stores the
        vocabulary and embedding matrix built during pre-processing and helps us prepare feature
        input for ner model; 2) model architecture, which describes the framework of our ner model;
        3) model weights, which stores the value of ner model's parameters.
        
        Args:
            preprocessor_file: path to save preprocessor
            json_file: path to save model architecture
            weights_file: path to save model weights, can be None. When we use `ModelCheckpoint` 
                          or `SWA` callback, model's weights will be saved to disk after training.
                          In that case, we don't need to save it again. We usually set weights_file
                          to be None.
        """
        self.preprocessor.save(preprocessor_file)
        logging.info('Save preprocessor to {}'.format(preprocessor_file))

        model_json = self.model.to_json()
        with open(json_file, 'w') as writer:
            writer.write(model_json)
        logging.info('Save model architecture to {}'.format(json_file))

        if weights_file:
            self.model.save_weights(weights_file)
            logging.info('Save model weights to {}'.format(weights_file))

    def load(self,
             preprocessor_file: str,
             json_file: str,
             weights_file: str,
             custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """Load ner application from disk.

        There are 3 things in total that we need to load: 1) preprocessor, which stores the
        vocabulary and embedding matrix built during pre-processing and helps us prepare feature
        input for ner model; 2) model architecture, which describes the framework of our ner model;
        3) model weights, which stores the value of ner model's parameters.

        Args:
            preprocessor_file: path to load preprocessor
            json_file: path to load model architecture
            weights_file: path to load model weights
            custom_objects: Optional dictionary mapping names (strings) to custom classes or
                            functions to be considered during deserialization. We will
                            automatically add all the custom layers of this project to
                            custom_objects. So you can ignore this argument in most cases unlesss
                            you use your own custom layer.

        """
        self.preprocessor = NERPreprocessor.load(preprocessor_file)
        logging.info('Load preprocessor from {}'.format(preprocessor_file))

        custom_objects = custom_objects or {}
        custom_objects.update(get_custom_objects())
        with open(json_file, 'r') as reader:
            self.model = tf.keras.models.model_from_json(reader.read(),
                                                         custom_objects=custom_objects)
        logging.info('Load model architecture from {}'.format(json_file))

        self.model.load_weights(weights_file)
        logging.info('Load model weight from {}'.format(weights_file))

        self.trainer = NERTrainer(self.model, self.preprocessor)
        self.predictor = NERPredictor(self.model, self.preprocessor)

    @staticmethod
    def get_ner_model(ner_model_type: str,
                      num_class: int,
                      use_char: bool,
                      char_embeddings: Optional[np.ndarray],
                      char_vocab_size: int,
                      char_embed_dim: int,
                      char_embed_trainable: bool,
                      use_bert: bool,
                      bert_config_file: Optional[str],
                      bert_checkpoint_file: Optional[str],
                      bert_trainable: bool,
                      use_word: bool,
                      word_embeddings: Optional[np.ndarray],
                      word_vocab_size: int,
                      word_embed_dim: int,
                      word_embed_trainable: bool,
                      max_len: Optional[int],
                      use_crf: bool,
                      optimizer: Union[str, tf.keras.optimizers.Optimizer],
                      **kwargs) -> tf.keras.models.Model:
        """Build ner model.

        Args:
            ner_model_type: str. Which type of ner model to use, can be one of {'bilstm',
                'bilstm-cnn', 'bigru', 'bigru-cnn', 'bert'}.
            num_class: int. Number of entity type. Usually calculated and passed by ner
                preprocessor.
            use_char: Boolean. Whether to use character embedding as input.
            char_embeddings: Optional np.ndarray. Char embedding matrix, shaped
                [char_vocab_size, char_embed_dim]. Usually pre-trained and passed by ner
                preprocessor. There are 2 cases when char_embeddings is None: 1)  use_char is
                False, do not use char embedding as input; 2) user did not provide valid
                pre-trained embedding file or any embedding training method.
            char_vocab_size: int. The size of char vocabulary. Usually calculated and passed by ner
                preprocessor.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            word_embeddings: Optional np.ndarray. Similar as char_embeddings.
            word_vocab_size: int. Similar as char_vocab_size.
            word_embed_dim: int. Similar as char_embed_dim.
            word_embed_trainable: Boolean. Similar as char_embed_trainable.
            max_len: Optional int, can be None. Max length of one sequence.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs: Other argument for building ner model, such as "rnn_units", "fc_dim" etc. See
                models.ner_models.py for there arguments.

        Raises:
            ValueError when `ner_model_type` not in one of {'bilstm', 'bilstm_cnn', 'bigru',
            'bigru_cnn', 'bert'}

        """

        if ner_model_type not in ner_model_dict:
            raise ValueError('`ner_model_type` not understood: {}'.format(ner_model_type))
        else:
            ner_model = ner_model_dict[ner_model_type](
                num_class=num_class,
                use_char=use_char,
                char_embeddings=char_embeddings,
                char_vocab_size=char_vocab_size,
                char_embed_dim=char_embed_dim,
                char_embed_trainable=char_embed_trainable,
                use_bert=use_bert,
                bert_config_file=bert_config_file,
                bert_checkpoint_file=bert_checkpoint_file,
                bert_trainable=bert_trainable,
                use_word=use_word,
                word_embeddings=word_embeddings,
                word_vocab_size=word_vocab_size,
                word_embed_dim=word_embed_dim,
                word_embed_trainable=word_embed_trainable,
                max_len=max_len,
                use_crf=use_crf,
                optimizer=optimizer,
                **kwargs
            )

            return ner_model.build_model()

    # todo: 重新训练模型
    def load_pretrained_model(self) -> None:
        """Load a pre-trained ner model that was trained on msra dataset using BiLSTM+CNN+CRF model.

        """
        cache_subdir = 'pretrained_models'

        prefix = 'https://fancy-nlp-1253403094.cos.ap-shanghai.myqcloud.com/pretrained_models/'

        preprocessor_file = tf.keras.utils.get_file(
            fname='msra_ner_bilstm_cnn_crf_preprocessor.pkl',
            origin=prefix+'msra_ner_bilstm_cnn_crf_preprocessor.pkl',
            cache_subdir=cache_subdir,
            cache_dir=CACHE_DIR)
        json_file = tf.keras.utils.get_file(
            fname='msra_ner_bilstm_cnn_crf.json',
            origin=prefix+'msra_ner_bilstm_cnn_crf.json',
            cache_subdir=cache_subdir,
            cache_dir=CACHE_DIR)
        weights_file = tf.keras.utils.get_file(
            fname='msra_ner_bilstm_cnn_crf.hdf5',
            origin=prefix+'msra_ner_bilstm_cnn_crf.hdf5',
            cache_subdir=cache_subdir,
            cache_dir=CACHE_DIR)

        self.load(preprocessor_file, json_file, weights_file)
