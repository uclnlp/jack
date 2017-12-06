# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install


def spacy_download_en():
    import subprocess
    args = ['python3 -m spacy download en']
    subprocess.call(args, shell=True)


def install_torch():
    import subprocess
    import sys
    if sys.version_info < (3, 6):
        args = ['pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl']
    else:
        args = ['pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl']
    subprocess.call(args, shell=True)


class Install(_install):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        _install.do_egg_install(self)
        spacy_download_en()
        install_torch()
        _install.run(self)


class Develop(_develop):
    def __init__(self):
        super().__init__()

    def run(self):
        spacy_download_en()
        install_torch()
        _develop.run(self)


with open('requirements.txt', 'r') as f:
    requirements = [l for l in f.readlines() if not l.startswith('http://')]

setup(name='jack',
      version='0.1.0',
      description='Jack the Reader is a Python framework for Machine Reading',
      author='UCL Machine Reading',
      author_email='s.riedel@cs.ucl.ac.uk',
      url='https://github.com/uclmr/jack',
      test_suite='tests',
      license='MIT',
      packages=find_packages(),
      cmdclass={
          'install': Install,
          'develop': Develop
      },
      install_requires=requirements,
      extras_require={
          'tensorflow': ['tensorflow>=1.4.0'],
          'tensorflow_gpu': ['tensorflow-gpu>=1.4.0'],
      },
      setup_requires=requirements,
      tests_require=requirements,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='tensorflow machine learning natural language processing question answering')
