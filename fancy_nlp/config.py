# -*- coding: utf-8 -*-

import os

CACHE_DIR = '~/.fancy_cache'
if not os.path.exists(os.path.expanduser(CACHE_DIR)):
    os.makedirs(os.path.expanduser(CACHE_DIR))
