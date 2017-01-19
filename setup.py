# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(name='jtr',
      version='0.1.0',
      description='Jack the Reader is a Python framework for Machine Reading',
      author='UCL Machine Reading',
      author_email='s.riedel@cs.ucl.ac.uk',
      url='https://github.com/uclmr/jtr',
      test_suite='tests',
      license='MIT',
      install_requires=[
            'tensorflow>=0.8',
            'pycorenlp>=0.3.0'
      ],
      packages=find_packages())
