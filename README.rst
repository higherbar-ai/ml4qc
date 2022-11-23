=====
ml4qc
=====

The ``ml4qc`` Python package offers a toolkit for employing machine learning technologies
in survey data quality control.

Installation
------------

Installing the latest version with pip::

    pip install ml4qc

Overview
--------

The ``ml4qc`` package builds on the `scikit-learn <https://scikit-learn.org/>`_ toolset. It includes the following
utility classes for working with survey data:

* ``SurveyML`` provides core functionality, including preprocessing and outlier detection
* ``SurveyMLClassifier`` builds on ``SurveyML``, adding support for running classification models and reporting out results

Examples
--------

This package is best illustrated by way of example. The following example analyses are available:

* `CATI1 analysis <https://github.com/orangechairlabs/ml4qc/blob/main/src/ml4qc-cati1-example.ipynb>`_

Documentation
-------------

See the full reference documentation here:

    https://ml4qc.readthedocs.io/

Development
-----------

To develop locally:

#. ``git clone https://github.com/orangechairlabs/ml4qc.git``
#. ``cd ml4qc``
#. ``python -m venv venv``
#. ``source venv/bin/activate``
#. ``pip install -r requirements.txt``

For convenience, the repo includes ``.idea`` project files for PyCharm.

To rebuild the documentation:

#. Update version number in ``/docs/source/conf.py``
#. Update layout or options as needed in ``/docs/source/index.rst``
#. In a terminal window, from the project directory:
    a. ``cd docs``
    b. ``SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -o source ../src/ml4qc --separate --force``
    c. ``make clean html``

To rebuild the distribution packages:

#. For the PyPI package:
    a. Update version number (and any build options) in ``/setup.py``
    b. Confirm credentials and settings in ``~/.pypirc``
    c. Run ``/setup.py`` for ``bdist_wheel`` build type (*Tools... Run setup.py task...* in PyCharm)
    d. Delete old builds from ``/dist``
    e. In a terminal window:
        i. ``twine upload dist/* --verbose``
#. For GitHub:
    a. Commit everything to GitHub and merge to ``main`` branch
    b. Add new release, linking to new tag like ``v#.#.#`` in main branch
#. For readthedocs.io:
    a. Go to https://readthedocs.org/projects/ml4qc/, log in, and click to rebuild from GitHub (only if it doesn't automatically trigger)
