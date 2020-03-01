# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from fancy_nlp.utils import load_ner_data_and_labels
from fancy_nlp.applications import NER

msra_train_file = 'datasets/ner/msra/train_data'
msra_dev_file = 'datasets/ner/msra/test_data'

checkpoint_dir = 'pretrained_models'
model_name = 'msra_ner_bert_crf'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

train_data, train_labels = load_ner_data_and_labels(msra_train_file)
valid_data, valid_labels = load_ner_data_and_labels(msra_dev_file)

ner = NER(use_pretrained=False)
ner.fit(train_data, train_labels, valid_data, valid_labels,
        ner_model_type='bert',
        use_char=False,
        use_word=False,
        use_bert=True,
        # 传入bert模型各文件的路径
        bert_vocab_file='pretrained_embeddings/chinese_L-12_H-768_A-12/vocab.txt',
        bert_config_file='pretrained_embeddings/chinese_L-12_H-768_A-12/bert_config.json',
        bert_checkpoint_file='pretrained_embeddings/chinese_L-12_H-768_A-12/bert_model.ckpt',
        bert_trainable=True,
        optimizer=tf.keras.optimizers.Adam(1e-5),
        callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        load_swa_model=True)

ner.save(preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
         json_file=os.path.join(checkpoint_dir, f'{model_name}.json'))

ner.load(preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
         json_file=os.path.join(checkpoint_dir, f'{model_name}.json'),
         weights_file=os.path.join(checkpoint_dir, f'{model_name}_swa.hdf5'))

print(ner.score(valid_data, valid_labels))
print(ner.analyze(train_data[2]))
print(ner.analyze_batch(train_data[:3]))
print(ner.restrict_analyze(train_data[2]))
print(ner.restrict_analyze_batch(train_data[:3]))
print(ner.analyze('同济大学位于上海市杨浦区，校长为陈杰'))
