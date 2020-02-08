# -*- coding: utf-8 -*-

from absl import logging
from keras.models import model_from_json
from keras.utils import get_file

from fancy_nlp.preprocessors import SPMPreprocessor
from fancy_nlp.models.spm import *
from fancy_nlp.trainers import SPMTrainer
from fancy_nlp.predictors import SPMPredictor
from fancy_nlp.utils import get_custom_objects
from fancy_nlp.config import CACHE_DIR, MODEL_STORAGE_PREFIX


class SPM(object):
    """SPM application"""

    def __init__(self, use_pretrained=False):
        self.preprocessor = None
        self.model = None
        self.trainer = None
        self.predictor = None

        if use_pretrained:
            self.load_pretrained_model()

    def fit(self,
            train_data,
            train_labels,
            valid_data=None,
            valid_labels=None,
            spm_model_type='siamese_cnn',
            use_word=True,
            external_word_dict=None,
            word_embed_type='word2vec',
            word_embed_dim=300,
            word_embed_trainable=True,
            use_char=False,
            char_embed_type='word2vec',
            char_embed_dim=300,
            char_embed_trainable=True,
            use_bert=False,
            bert_vocab_file=None,
            bert_config_file=None,
            bert_checkpoint_file=None,
            bert_trainable=False,
            label_dict_file=None,
            max_len=None,
            max_word_len=None,
            optimizer='adam',
            batch_size=32,
            epochs=50,
            callback_list=None,
            checkpoint_dir=None,
            model_name=None,
            load_swa_model=False,
            **kwargs):
        """Train spm model using provided data

        Args:
            train_data: list of untokenized text pairs for training,
                        like ``[['我是中国人', ...], ['我爱中国', ...]]``
            train_labels: labels string of train_data
            valid_data: list of untokenized text pairs for evaluation
            valid_labels: labels string of valid data
            spm_model_type: str, which spm model to use
            use_word: boolean, whether to use word embedding as input
            external_word_dict: external word dictionary
            word_embed_type: str, can be a pre-trained embedding filename or pre-trained embedding
                             methods (word2vec, glove, fastext)
            word_embed_dim: int, dimensionality of word embedding
            word_embed_trainable: boolean, whether to update word embedding during training
            use_char: boolean, whether to use char as input
            char_embed_type: similar as 'word_embed_type'
            char_embed_dim: similar as 'word_embed_dim'
            char_embed_trainable: similar as 'word_embed_trainable'
            use_bert: boolean, whether to use bert embedding as input
            bert_vocab_file: str, path to bert's vocabulary file
            bert_config_file: str, path to bert's configuration file
            bert_checkpoint_file: str, path to bert's checkpoint file
            bert_trainable: boolean, whether to update bert during training
            use_bert_model: boolean, whether to use traditional bert model which combines two sentences
                            as one input
            label_dict_file: a file with two columns separated by tab, the first column is raw
                             label name, and the second column is the corresponding name which is
                             meaningful
            max_len: int, max sequence length. If None, we dynamically use the max length of one batch
                     as max_len. However, max_len must be provided when using bert as input.
            max_word_len: int max word length. If None, we dynamically use the max word length of one
                          batch as max_word_len.
            optimizer: str or instance of `keras.optimizers.Optimizer`, indicating the optimizer to
                       use during training
            batch_size: num of samples per gradient update
            epochs: num of epochs to train the model
            callback_list: list of str, each item indicates the callback to apply during training
                           Currently, we support using 'modelcheckpoint' for `ModelCheckpoint`
                           callback, 'earlystopping` for 'Earlystopping` callback, 'swa' for
                           'SWA' callback. We will automatically add `SPMMetric` callback when
                           valid_data and valid_labels are both provided.
            checkpoint_dir: str, directory to save spm model, must be provided when using
                            `ModelCheckpoint` or `SWA` callback.
            model_name: str, prefix of spm model's weights file must be provided when using
                        `ModelCheckpoint` or `SWA` callback.
                        For example, if checkpoint_dir is 'ckpt' and model_name is 'model', the
                        weights of spm model saved by `ModelCheckpoint` callback will be
                        'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'
            load_swa_model: boolean, whether to load swa model, only apply when using SWA Callback.
            **kwargs: other argument for building spm model, such as "rnn_units", "fc_dim" etc.
        """
        use_bert_model = True if spm_model_type == 'bert' else False

        self.preprocessor = SPMPreprocessor(train_data=train_data,
                                            train_labels=train_labels,
                                            use_word=use_word,
                                            use_char=use_char,
                                            use_bert=use_bert,
                                            use_bert_model=use_bert_model,
                                            external_word_dict=external_word_dict,
                                            bert_vocab_file=bert_vocab_file,
                                            char_embed_type=char_embed_type,
                                            char_embed_dim=char_embed_dim,
                                            word_embed_type=word_embed_type,
                                            word_embed_dim=word_embed_dim,
                                            label_dict_file=label_dict_file,
                                            max_len=max_len,
                                            max_word_len=max_word_len)

        self.model = self.get_spm_model(spm_model_type=spm_model_type,
                                        num_class=self.preprocessor.num_class,
                                        use_word=use_word,
                                        word_embeddings=self.preprocessor.word_embeddings,
                                        word_vocab_size=self.preprocessor.word_vocab_size,
                                        word_embed_dim=self.preprocessor.word_embed_dim,
                                        word_embed_trainable=word_embed_trainable,
                                        use_char=use_char,
                                        char_embeddings=self.preprocessor.char_embeddings,
                                        char_vocab_size=self.preprocessor.char_vocab_size,
                                        char_embed_dim=self.preprocessor.char_embed_dim,
                                        char_embed_trainable=char_embed_trainable,
                                        use_bert=use_bert,
                                        bert_config_file=bert_config_file,
                                        bert_checkpoint_file=bert_checkpoint_file,
                                        bert_trainable=bert_trainable,
                                        max_len=self.preprocessor.max_len,
                                        max_word_len=self.preprocessor.max_word_len,
                                        optimizer=optimizer,
                                        **kwargs)

        if 'swa' in callback_list:
            swa_model = self.get_spm_model(spm_model_type=spm_model_type,
                                           num_class=self.preprocessor.num_class,
                                           use_word=use_word,
                                           word_embeddings=self.preprocessor.word_embeddings,
                                           word_vocab_size=self.preprocessor.word_vocab_size,
                                           word_embed_dim=self.preprocessor.word_embed_dim,
                                           word_embed_trainable=word_embed_trainable,
                                           use_char=use_char,
                                           char_embeddings=self.preprocessor.char_embeddings,
                                           char_vocab_size=self.preprocessor.char_vocab_size,
                                           char_embed_dim=self.preprocessor.char_embed_dim,
                                           char_embed_trainable=char_embed_trainable,
                                           use_bert=use_bert,
                                           bert_config_file=bert_config_file,
                                           bert_checkpoint_file=bert_checkpoint_file,
                                           bert_trainable=bert_trainable,
                                           max_len=self.preprocessor.max_len,
                                           max_word_len=self.preprocessor.max_word_len,
                                           optimizer=optimizer,
                                           **kwargs)
        else:
            swa_model = None

        self.trainer = SPMTrainer(self.model, self.preprocessor)
        self.trainer.train_generator(train_data, train_labels, valid_data, valid_labels,
                                     batch_size, epochs, callback_list, checkpoint_dir, model_name,
                                     swa_model, load_swa_model)

        self.predictor = SPMPredictor(self.model, self.preprocessor)

        if valid_data is not None and valid_labels is not None:
            logging.info('Evaluating on validation data...')
            self.score(valid_data, valid_labels)

    def score(self, valid_data, valid_labels):
        """Return the f1 score of the model over validation data

        Args:
            valid_data: list of untokenized text pairs
            valid_labels: list of label strings

        Returns:

        """
        if self.trainer:
            return self.trainer.evaluate(valid_data, valid_labels)
        else:
            logging.fatal('Trainer is None! Call fit() or load() to get trainer.')

    def predict(self, test_text):
        """Return prediction of the model for test data

        Args:
            test_text: a pair of untokenized text

        Returns:

        """
        if self.predictor:
            return self.predictor.matching(test_text)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def predict_batch(self, test_texts):
        """Return predictions of the model for test data

        Args:
            test_texts: list of untokenized text pairs

        Returns:

        """
        if self.predictor:
            return self.predictor.matching_batch(test_texts)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def analyze(self, text):
        """Analyze text and return matching result with probability.

        Args:
            text: a pair of untokenized text
        Returns:

        """
        if self.predictor:
            return self.predictor.matching_with_prob(text)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def analyze_batch(self, texts):
        """Analyze text and return matching result with probability.

        Args:
            texts: list of untokenized text pairs
        Returns:

        """
        if self.predictor:
            return self.predictor.matching_with_prob_batch(texts)
        else:
            logging.fatal('Predictor is None! Call fit() or load() to get predictor.')

    def save(self, preprocessor_file, json_file, weights_file=None):
        """save spm application

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

    def load(self, preprocessor_file, json_file, weights_file, custom_objects=None):
        """load spm application

        Args:
            preprocessor_file: path to load preprocessor
            json_file: path to load model architecture
            weights_file: path to load model weights
            custom_objects: Optional dictionary mapping names (strings) to custom classes or
                            functions to be considered during deserialization. Must provided when
                            using custom layer.

        """
        self.preprocessor = SPMPreprocessor.load(preprocessor_file)
        logging.info('Load preprocessor from {}'.format(preprocessor_file))

        custom_objects = custom_objects or {}
        custom_objects.update(get_custom_objects())
        with open(json_file, 'r') as reader:
            self.model = model_from_json(reader.read(), custom_objects=custom_objects)
        logging.info('Load model architecture from {}'.format(json_file))

        self.model.load_weights(weights_file)
        logging.info('Load model weight from {}'.format(weights_file))

        self.trainer = SPMTrainer(self.model, self.preprocessor)
        self.predictor = SPMPredictor(self.model, self.preprocessor)

    @staticmethod
    def get_spm_model(spm_model_type, num_class, use_word, word_embeddings, word_vocab_size,
                      word_embed_dim, word_embed_trainable, use_char, char_embeddings,
                      char_vocab_size, char_embed_dim, char_embed_trainable, use_bert,
                      bert_config_file, bert_checkpoint_file, bert_trainable,
                      max_len, max_word_len, optimizer, **kwargs):
        spm_model_all = {'siamese_cnn': SiameseCNN,
                         'siamese_bilstm': SiameseBiLSTM,
                         'siamese_bigru': SiameseBiGRU,
                         'esim': ESIM,
                         'bimpm': BiMPM,
                         'bert': BertSPM}
        if spm_model_type in spm_model_all:
            spm_model = spm_model_all[spm_model_type](
                num_class=num_class,
                use_word=use_word,
                word_embeddings=word_embeddings,
                word_vocab_size=word_vocab_size,
                word_embed_dim=word_embed_dim,
                word_embed_trainable=word_embed_trainable,
                use_char=use_char,
                char_embeddings=char_embeddings,
                char_vocab_size=char_vocab_size,
                char_embed_dim=char_embed_dim,
                char_embed_trainable=char_embed_trainable,
                use_bert=use_bert,
                bert_config_file=bert_config_file,
                bert_checkpoint_file=bert_checkpoint_file,
                bert_trainable=bert_trainable,
                max_len=max_len,
                max_word_len=max_word_len,
                optimizer=optimizer,
                **kwargs)

        else:
            raise ValueError('`spm_model_type` not understood: {}'.format(spm_model_type))

        return spm_model.build_model()

    # todo: 重新训练模型
    def load_pretrained_model(self):
        cache_subdir = 'pretrained_models'

        preprocessor_file = get_file(
            fname='webank_spm_siamese_cnn_word_preprocessor.pkl',
            origin=MODEL_STORAGE_PREFIX + 'webank_spm_siamese_cnn_word_preprocessor.pkl',
            cache_subdir=cache_subdir, cache_dir=CACHE_DIR)
        json_file = get_file(
            fname='webank_spm_siamese_cnn_word.json',
            origin=MODEL_STORAGE_PREFIX + 'webank_spm_siamese_cnn_word.json',
            cache_subdir=cache_subdir, cache_dir=CACHE_DIR)
        weights_file = get_file(
            fname='webank_spm_siamese_cnn_word.hdf5',
            origin=MODEL_STORAGE_PREFIX + 'webank_spm_siamese_cnn_word.hdf5',
            cache_subdir=cache_subdir, cache_dir=CACHE_DIR)

        self.load(preprocessor_file, json_file, weights_file)
