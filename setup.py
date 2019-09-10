# -*- coding: utf-8 -*-
import codecs

from setuptools import setup
from setuptools import find_packages

long_description = '''
fancy-nlp is a fast and easy-to-use natural language processing (NLP) toolkit,
satisfying your imagination about NLP.

fancy-nlp is compatible with Python 3.6
and is distributed under the GPLv3 license.
'''

with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name='fancy-nlp',
      version='0.0.2',
      author='boat-group',
      author_email='e.shijia@foxmail.com',
      description='NLP for humans',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/boat-group/fancy-nlp",
      install_requires=install_requires,
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())

