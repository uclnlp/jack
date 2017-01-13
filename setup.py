# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(name='quebap',
      version='0.1.0',
      description='QUEstion answering and knowledge BAse Population',
      author='UCL Machine Reading',
      author_email='s.riedel@cs.ucl.ac.uk',
      url='https://github.com/uclmr/quebap',
      test_suite='tests',
      license='MIT',
      install_requires=[
            'tensorflow>=0.8',
            'pycorenlp>=0.3.0'
      ],
      packages=find_packages())
