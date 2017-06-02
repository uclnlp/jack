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


with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(name='jtr',
      version='0.1.0',
      description='Jack the Reader is a Python framework for Machine Reading',
      author='UCL Machine Reading',
      author_email='s.riedel@cs.ucl.ac.uk',
      url='https://github.com/uclmr/jtr',
      test_suite='tests',
      license='MIT',
      cmdclass={'install': Install},
      install_requires=requirements,
      setup_requires=['pytest-runner'] + requirements,
      tests_require=requirements,
      packages=find_packages())
