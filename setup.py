# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install


def spacy_download_en():
    import spacy
    try:
        spacy.load('en')
    except RuntimeError:
        import subprocess
        args = ['python3 -m spacy download en']
        subprocess.call(args, shell=True)


class Install(_install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        _install.do_egg_install(self)
        spacy_download_en()
        _install.run(self)


class Develop(_develop):
    def __init__(self):
        super().__init__()

    def run(self):
        spacy_download_en()
        _develop.run(self)


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
      cmdclass={'install': Install, 'develop': Develop},
      install_requires=requirements,
      extras_require={
          'gpu': ['tensorflow-gpu>=1.3']
      },
      setup_requires=['pytest-runner'] + requirements,
      tests_require=requirements,
      packages=find_packages())
