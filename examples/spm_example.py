# -*- coding: utf-8 -*-

import os

from fancy_nlp.utils import load_spm_data_and_labels
from fancy_nlp.applications import SPM

train_file = 'datasets/spm/webank/BQ_train.txt'
valid_file = 'datasets/spm/webank/BQ_dev.txt'
test_file = 'datasets/spm/webank/BQ_test.txt'

model_name = 'webank_spm_siamese_cnn_word'
checkpoint_dir = 'pretrained_models'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

train_data, train_labels = load_spm_data_and_labels(train_file)
valid_data, valid_labels = load_spm_data_and_labels(valid_file)
test_data, test_labels = load_spm_data_and_labels(test_file)

spm_app = SPM(use_pretrained=False)

spm_app.fit(train_data, train_labels, valid_data, valid_labels,
            spm_model_type='siamese_cnn',
            word_embed_trainable=True,
            callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            max_len=60,
            load_swa_model=True)

spm_app.save(
    preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
    json_file=os.path.join(checkpoint_dir, f'{model_name}.json'))

spm_app.load(
    preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
    json_file=os.path.join(checkpoint_dir, f'{model_name}.json'),
    weights_file=os.path.join(checkpoint_dir, f'{model_name}_swa.hdf5'))

print(spm_app.score(test_data, test_labels))
print(spm_app.predict(('未满足微众银行审批是什么意思', '为什么我未满足微众银行审批')))
print(spm_app.analyze(('未满足微众银行审批是什么意思', '为什么我未满足微众银行审批')))
