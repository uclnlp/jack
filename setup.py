# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install as _install


class Install(_install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')


with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(name='jack',
      version='0.1.0',
      description='Jack the Reader is a Python framework for Machine Reading',
      author='UCL Machine Reading',
      author_email='s.riedel@cs.ucl.ac.uk',
      url='https://github.com/uclmr/jack',
      test_suite='tests',
      license='MIT',
      cmdclass={'install': Install, 'develop': Install},
      install_requires=requirements,
      setup_requires=['pytest-runner'] + requirements,
      tests_require=requirements,
      packages=find_packages())
