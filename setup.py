#!/usr/bin/env python
from distutils.core import setup


with open('requirements.txt', 'r') as reqf:
    required = reqf.read().splitlines()


setup(
    name='music2vec',
    version='0.1',
    description='Music to vector encoding',
    author='Dmitry Shmatkov',
    packages=['common', 'model', 'preprocess_dataset'],
    install_requires=required,
    scripts=['cli'],
)
