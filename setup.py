
import os
from setuptools import setup

setup(
    name = "ensembles",
    version = "0.1.5",
    author = "Prajwal Kailas",
    author_email = "prajwal967@gmail.com",
    description = ("A package for ensembling of machine learning models"),
    license = "MIT",
    keywords = "Ensemble",
    url = "https://github.com/unnati-xyz/ensemble-package",
    packages=['ensembles'],
    classifiers=['Development Status :: 3 - Alpha',

     'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.6',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',],
	install_requires=[
   	'pandas',
   	'numpy',
	'xgboost',
	'category_encoders',
	'sklearn',
	'keras',
	'joblib',
	'hyperopt']
)

