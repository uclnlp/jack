# How to Generate New Docs From Scratch
Install [Sphinx](http://www.sphinx-doc.org/en/1.5.1/install.html) and its requirements. Then execute the steps below as taken from [this github issue](Ahttp://stackoverflow.com/questions/20354768/python-sphinx-how-to-document-one-file-with-functions)

Here is a step-by-step list:

1. Create documentation folder: `mkdir doc`
2. Enter doc/: `cd doc`
3. Execute sphinx-quickstart (Be sure to select autodoc: y, Makefile: y)
4. Edit conf.py to specify sys.path: `sys.path.insert(0, os.path.abspath('..'))`
5. Edit index.rst and specify modules in the toctree:
```
.. toctree::
    :maxdepth: 2

    modules
```
6. Execute sphinx-apidoc -o . ..
7. Generate the html output: make html
8. View your documentation: firefox _build/html/index.html
