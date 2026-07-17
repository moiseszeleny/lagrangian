"""Sphinx configuration for the feynlag documentation site."""

project = "feynlag"
copyright = "2026, Moises Zeleny"
author = "Moises Zeleny"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# .rst must be the FIRST key: Sphinx's autosummary get_rst_suffix() walks
# this dict in order and (due to a quirk in how it looks up registered
# parsers by raw suffix text) treats any suffix it doesn't recognize as
# "supports restructuredtext" too -- so an unmapped .md ahead of .rst would
# make autosummary write its generated ".. automodule::" stub pages with a
# .md extension, and MyST would then render those RST directives as
# literal text instead of executing them.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

# Notebooks are committed already-executed (nbstripout --keep-output); the
# doc build renders stored outputs and never re-runs the physics.
nb_execution_mode = "off"

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The tutorial notebooks are symlinked in from examples/ and contain
# relative markdown links (../CONVENTIONS.md, sm_scalar_gauge.py, ...)
# written for their original location; those resolve fine on GitHub but not
# from docs/tutorials/, and editing the notebooks themselves is out of
# scope (CLAUDE.md: notebooks are tracked pre-executed, edit only via
# nbconvert --execute). Suppress just this warning class rather than -W
# the whole build on noise we can't fix without touching notebook content.
suppress_warnings = ["myst.xref_missing"]

html_theme = "furo"
html_title = "feynlag"
myst_heading_anchors = 3
