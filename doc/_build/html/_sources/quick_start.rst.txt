#####################################
Quick Start with scikit-dimension
#####################################

This package provides Python implementations of intrinsic dimension estimators.

Creating your own scikit-learn contribution package
===================================================

1. Installation
-------------------------------------

To install with pip run::

    $ git clone https://github.com/scikit-learn-contrib/project-template.git

To install from source run:

4. Setup the continuous integration
-----------------------------------

The project template already contains configuration files of the continuous
integration system. Basically, the following systems are set:

* Travis CI is used to test the package in Linux. You need to activate Travis
  CI for your own repository. Refer to the Travis CI documentation.
* AppVeyor is used to test the package in Windows. You need to activate
  AppVeyor for your own repository. Refer to the AppVeyor documentation.
* Circle CI is used to check if the documentation is generated properly. You
  need to activate Circle CI for your own repository. Refer to the Circle CI
  documentation.
* ReadTheDocs is used to build and host the documentation. You need to activate
  ReadTheDocs for your own repository. Refer to the ReadTheDocs documentation.
* CodeCov for tracking the code coverage of the package. You need to activate
  CodeCov for you own repository.
* PEP8Speaks for automatically checking the PEP8 compliance of your project for
  each Pull Request.

Publish your package
====================

.. _PyPi: https://packaging.python.org/tutorials/packaging-projects/
.. _conda-foge: https://conda-forge.org/

You can make your package available through PyPi_ and conda-forge_. Refer to
the associated documentation to be able to upload your packages such that
it will be installable with ``pip`` and ``conda``. Once published, it will
be possible to install your package with the following commands::

    $ pip install your-scikit-learn-contribution
    $ conda install -c conda-forge your-scikit-learn-contribution
