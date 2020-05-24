# QBUS6850_2020_S1

## sklearn compatibility checks

To check your implementation do:

    from util import check_estimator_adaboost
    from AdaBoost import AdaBoostClassifier

    check_estimator_adaboost(AdaBoostClassifier)

See  `demo.py` script for an example.

## PEP8 formatting

To check PEP8 formatting do:

    flake8 AdaBoost.py
    
To install flake8 do:

    conda install -c anaconda flake8

## documentation

To check your documentation style do:

    pydocstyle --convention=numpy AdaBoost.py

To install pydocstyle do:

    conda install -c conda-forge pydocstyle


