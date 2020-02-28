# -*- coding: utf-8 -*-

import os

from fancy_nlp.utils import load_text_classification_data_and_labels
from fancy_nlp.applications import TextClassification

data_file = 'datasets/text_classification/toutiao/toutiao_cat_data.txt'
dict_file = 'datasets/text_classification/toutiao/toutiao_label_dict.txt'
model_name = 'toutiao_text_classification_cnn'
checkpoint_dir = 'pretrained_models'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

train_data, train_labels, valid_data, valid_labels, test_data, test_labels = \
    load_text_classification_data_and_labels(data_file,
                                             label_index=1,
                                             text_index=3,
                                             delimiter='_!_',
                                             split_mode=2,
                                             split_size=0.3)

text_classification_app = TextClassification(use_pretrained=False)

text_classification_app.fit(train_data, train_labels, valid_data, valid_labels,
                            text_classification_model_type='cnn',
                            char_embed_trainable=True,
                            callback_list=['modelcheckpoint', 'earlystopping', 'swa'],
                            checkpoint_dir=checkpoint_dir,
                            model_name=model_name,
                            label_dict_file=dict_file,
                            max_len=60,
                            load_swa_model=True)

# noinspection DuplicatedCode
text_classification_app.save(
    preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
    json_file=os.path.join(checkpoint_dir, f'{model_name}.json'))

# noinspection DuplicatedCode
text_classification_app.load(
    preprocessor_file=os.path.join(checkpoint_dir, f'{model_name}_preprocessor.pkl'),
    json_file=os.path.join(checkpoint_dir, f'{model_name}.json'),
    weights_file=os.path.join(checkpoint_dir, f'{model_name}_swa.hdf5'))

print(text_classification_app.score(test_data, test_labels))
print(text_classification_app.predict('小米公司成立十周年'))
print(text_classification_app.analyze('小米公司成立十周年'))
