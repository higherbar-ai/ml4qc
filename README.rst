=====
ml4qc
=====

The ``ml4qc`` Python package offers a toolkit for employing machine learning technologies
in survey data quality control. Among other things, it helps to extend
`the surveydata package <https://github.com/orangechairlabs/py-surveydata>`_ and advance `SurveyCTO's
machine learning roadmap <https://www.surveycto.com/blog/machine-learning-for-quality-control/>`_.
See further below for more about the research program and specifics on the package.

Installation
------------

Installing the latest version with pip::

    pip install ml4qc

Overview of research program
----------------------------

The working title and abstract for the overarching research program is as follows:

    **Can machine learning aid survey data quality-control efforts, even without access to actual
    survey data?**

    A robust quality-control process with some degree of human review is often crucial for survey
    data quality, but resources for human review are almost always limited and therefore rationed.
    While traditional statistical methods of directing quality-control efforts often rely on
    field-by-field analysis to check for outliers, enumerator effects, and unexpected patterns,
    newer machine-learning-based methods allow for a more holistic evaluation of interviews. ML
    methods also allow for human review to train models that can then predict the results of
    review, increasingly focusing review time on potential problems. In this research program, we
    explore the potential of ML-based methods to direct and supplement QC efforts across
    international settings. In particular, we look at the potential for privacy-protecting
    approaches that allow ML models to be trained and utilized without ever exposing
    personally-identifiable data — or, indeed, any survey data at all — to ML systems or analysts.
    Specifically, metadata and paradata, including rich but non-identifying data from mobile device
    sensors, is used in lieu of potentially-sensitive survey data.

We currently envision three phases to the work:

* Phase 1: Foundation-building and early exploration
    * **Goal 1**: Establish a flexible codebase to serve as a platform for exploration and analysis
    * **Goal 2**: Test whether popular supervised prediction models work to predict quality classifications when trained
      on meta/paradata
    * **Goal 3**: Test whether popular unsupervised models can identify useful patterns in enumerator effects
* Phase 2: Dig deeper across settings, refine tools and understanding
    * **Goal 1**: Tune models and features for supervised models, establish links between review processes, quality
      classifications, and effectiveness
    * **Goal 2**: Find useful structure for measuring and reporting enumerator effects, how to control out nonrandom
      sources of variation
    * **Goal 3**: For all models, establish requirements for training data (e.g., sample sizes)
* Phase 3: Test the potential for transfer learning, further refine tools
    * **Goal 1**: Develop instrument-agnostic meta/paradata format and test the potential for transfer learning to
      enable useful results earlier in the launch of a new survey
    * **Goal 2**: Support continued scaling of experimentation across more settings with easy, production-ready tools

We are currently in Phase 1. See the "Examples" section below for early results.

Overview of Python package
--------------------------

The ``ml4qc`` package builds on the `scikit-learn <https://scikit-learn.org/>`_ toolset. It includes the following
utility classes for working with survey data:

* ``SurveyML`` provides core functionality, including preprocessing, outlier detection, and cluster analysis
* ``SurveyMLClassifier`` builds on ``SurveyML``, adding support for running classification models and reporting out
  results

While ``SurveyMLClassifier`` supports a variety of approaches, the currently-recommended
approach to binary classification is as follows:

1. Do *not* reweight for class imbalances; use
   ``SurveyMLClassifier.cv_for_best_hyperparameters()`` to find the optimal hyperparameters
   for a given dataset, with *neg_log_loss*, *neg_brier_score*, or *roc_auc* as the CV metric
   to optimize. This will optimize for an unbiased distribution of estimated probabilities.
2. Use a ``calibration_method`` (*isotonic* or *sigmoid*) to calibrate the estimated
   probability distribution.
3. Almost always (and especially when classes are imbalanced), specify a non-default option
   for the classification ``threshold`` (and possibly ``threshold_value``), as the
   default threshold of 0.5 is unlikely to be optimal. When in doubt, use
   ``threshold='optimal_f'`` to choose the threshold that maximizes the F-1 score.

This is essentially the approach used in the examples linked below.

When there are nonrandom aspects to interview assignment, it is also recommended to initialize ``SurveyMLClassifier``
with a list of ``control_features`` to control out as much of the nonrandom-assignment effects as possible. During
pre-processing, ``control_features`` will be used in OLS regressions to predict each other feature value, and it will
be the residuals that are used in further analysis. This can be particularly important in distinguishing enumerator
effects from the effects of nonrandom assignment. See the ``CAPI2`` example linked below.

Examples
--------

This package is best illustrated by way of example. The following example analyses are available:

* `CATI1 analysis <https://github.com/orangechairlabs/ml4qc/blob/main/src/ml4qc-cati1-example.ipynb>`_
    * **Setting**: CATI survey in Afghanistan
    * **Review and quality classification strategy**: Holistic review of all submissions
    * **Supervised results**: Precision for predicting rejected submission ranges from 10% to 43% (against a base rate of
      3.8%)
    * **Unsupervised results**: Significant enumerator effects discovered and summarized
* `CATI2 analysis <https://github.com/orangechairlabs/ml4qc/blob/main/src/ml4qc-cati2-example.ipynb>`_
    * **Setting**: CATI survey in Afghanistan
    * **Review and quality classification strategy**: Holistic review of all submissions
    * **Supervised results**: Precision for predicting rejected submission ranges from 20-50% (against a base rate of 4.7%),
      but wide variation due to very small training sample
    * **Unsupervised results**: Significant enumerator effects discovered and summarized, but not at cluster level
* `CATI3 analysis <https://github.com/orangechairlabs/ml4qc/blob/main/src/ml4qc-cati3-example.ipynb>`_
    * **Setting**: CATI survey in Afghanistan
    * **Review and quality classification strategy**: All completed interviews classified the same, all incomplete
      interviews rejected
    * **Supervised results**: Because there are clear meta/paradata differences between complete and incomplete interviews,
      all ML models achieve 100% precision in predicting rejection
    * **Unsupervised results**: Very significant enumerator effects discovered and summarized
* `CAPI1 analysis <https://github.com/orangechairlabs/ml4qc/blob/main/src/ml4qc-capi1-example.ipynb>`_
    * **Setting**: CAPI survey in Afghanistan
    * **Review and quality classification strategy**: Holistic review of all submissions
    * **Supervised results**: With only 5 rejected submissions, instead sought to predict "not approved as GOOD quality”
      with a base rate of 70% (resting almost completely on the distinction between a “GOOD” and an “OKAY” quality
      rating); none of the models succeed in predicting the distinction and it's not clear that a larger sample size
      would help
    * **Unsupervised results**: Very significant enumerator effects discovered and summarized
* `CAPI2 analysis <https://github.com/orangechairlabs/ml4qc/blob/main/src/ml4qc-capi2-example.ipynb>`_
    * **Setting**: CAPI survey in Ethiopia
    * **Review and quality classification strategy**: Submissions flagged with automated statistical checks at the question
      level, plus randomly-selected interviews, reviewed for individual responses in need of correction; those that
      require correction classified as "OKAY" (vs. "GOOD") quality
    * **Supervised results**: Full results still TBD, but predictive results poor overall, though slightly better with
      structural models (logistic regression and neural networks)
    * **Unsupervised results**: Once many of the effects of nonrandom assignment are controlled out, there are only
      weak enumerator effects at the cluster level

Documentation
-------------

See the full reference documentation here:

    https://ml4qc.readthedocs.io/

Project support
---------------

`Dobility <https://www.surveycto.com/>`_ has generously provided financial and other support for v1 of the ``ml4qc``
package, including support for early testing and piloting.

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
