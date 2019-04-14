.. image:: https://github.com/lijunzh/yews/blob/master/logo/logo.gif
   :scale: 50 %
   :alt: Yews Logo
   :align: center


Yews | Your Earthquake Waveform Solution
========================================


.. image:: https://travis-ci.com/lijunzh/yews.svg?branch=master
    :target: https://travis-ci.com/lijunzh/yews

.. image:: https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva?svg=true
    :target: https://ci.appveyor.com/project/lijunzh/yews

.. image:: https://codecov.io/gh/lijunzh/yews/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/lijunzh/yews

.. image:: https://anaconda.org/lijunzhu/yews/badges/version.svg   
    :target: https://anaconda.org/lijunzhu/yews

.. image:: https://badge.fury.io/py/yews.svg
    :target: https://badge.fury.io/py/yews

.. image:: https://pepy.tech/badge/yews
    :target: https://pepy.tech/project/yews


Yews is a deep learning toolbox for processing seismic waveform made with flexibility, speed, and usability in mind. It is built upon `PyTorch <https://github.com/pytorch/pytorch>`_ for researchers interested in applying deep learning techniques on seismic waveform data.



Installation
============

First install `PyTorch <https://github.com/pytorch/pytorch>`_ using the offical guide: https://pytorch.org/get-started/locally/ . Then, install `Yews <https://github.com/lijunzh/yews>`_ via one of the following approaches:

conda:

.. code:: bash

   conda install yews -c lijunzhu


pip:

.. code:: bash

   pip install yews


From source:

.. code:: bash

    python setup.py install


Note:

1. These assume that you have `PyTorch <https://pytorch.org/get-started/locally/>`_ installed via the default method.

2. Yews can be installed via *conda-forge* with all dependencies handled automatically; however, it only supports the **CPU** version as *PyTorch* does not have a **GPU** version for the conda-forge channel.



Documentation
=============

Under construction.

Contributing
============

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

