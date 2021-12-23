# How to generate the doc.

1. Install the `sphinx` and  it's extensions.

```
pip install sphinx
pip install myst-parser -- for surport markdown
pip install recommonmark -- for surport markdown
pip install sphinx_rtd_theme --read_the_docs theme
pip install sphinx-markdown-checkbox
```

2. Generate from template

```
dlt/docs$ sphinx-quickstart
    Separate source and build directories(y/n) : y

```

3. Update the docs/source/conf.py

```
from recommonmark.transform import AutoStructify
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../dlk'))

github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/doc/'
def setup(app):
    app.add_config_value('recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)


# -- Project information -----------------------------------------------------

project = 'dlk'
copyright = '2021, cstsunfu'
author = 'cstsunfu'

# The full version, including alpha/beta/rc tags
release = 'v0.0.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # for markdown
    'sphinx.ext.autodoc', 'sphinx.ext.viewcode', # for source code view
    'sphinx.ext.napoleon', # for google style doc
    'sphinx.ext.todo', # for todo
    'sphinx.ext.mathjax', # for latex
    'sphinx_markdown_checkbox', # for markdown checkbox
]

source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}
source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme' # use the read_the_docs theme

html_theme_options = {
    "collapse_navigation" : False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
```

4. Gather the doc to .rst
```
dlk/docs $ sphinx-apidoc -o ./source ../dlk/

```

5. Add Links

```

docs/source $ ln -s ../../README.md introduction.md
docs/source $ ln -s ../../dlk/data/processors/readme.md process_progress.md
docs/source $ ln -s ../../dlk/readme.md appointment.md
```

6. update index
```
update the source/index.rst


Deep Learning ToolKit
=====================

.. toctree::
   :maxdepth: 1
   :name: Introduction
   :caption: Introduction

   introduction

.. toctree::
   :name: api
   :caption: API References
   :maxdepth: 3

   dlk.core
   dlk.data
   dlk.managers
   dlk.utils


.. toctree::
   :maxdepth: 1
   :name: Appointments
   :caption: Appointments

   appointment

.. toctree::
   :maxdepth: 1
   :name: processor
   :caption: Process Progress

   process_progress

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

7. make html
```
dlk/docs $ make html

```

8. view docs
```
open build/html/index.html
```
