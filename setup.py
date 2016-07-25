
import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ensemble",
    version = "0.2.1",
    author = "Prajwal Kailas",
    author_email = "prajwal967@gmail.com",
    description = ("A package for ensembling of machine learning models"),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "https://github.com/unnati-xyz/ensemble-package",
    packages=['ensemble'],
    long_description=read('README.md'),
    classifiers=[]
)

