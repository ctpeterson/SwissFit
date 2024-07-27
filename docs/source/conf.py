# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# I got a lot of help from: https://pennyhow.github.io/blog/making-readthedocs/

import sys
import os
import sphinx_pdj_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SwissFit'
copyright = '2024, Curtis Taylor Peterson'
author = 'Curtis Taylor Peterson'
release = '0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon'
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_pdj_theme'
html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

# -- Paths for autodoc generation --------------------------------------------
sys.path.insert(0,os.path.abspath('../../src/swissfit/'))
sys.path.insert(0,os.path.abspath('.'))
