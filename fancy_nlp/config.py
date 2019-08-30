# -*- coding: utf-8 -*-

import os

CACHE_DIR = '~/.fancy_cache'
if not os.path.exists(os.path.expanduser(CACHE_DIR)):
    os.makedirs(os.path.expanduser(CACHE_DIR))

MODEL_STORAGE_PREFIX = \
    'https://fancy-nlp-1253403094.cos.ap-shanghai.myqcloud.com/pretrained_models/'
