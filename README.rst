.. image:: https://raw.githubusercontent.com/lijunzh/yews-docs/master/source/_static/img/logo/yews_logo.gif
   :scale: 50 %
   :alt: Yews Logo
   :align: center


========================================



Yews is a deep learning toolbox for processing seismic waveform made with
flexibility, speed, and usability in mind. It is built upon
`PyTorch <https://github.com/pytorch/pytorch>`_ for researchers interested in
applying deep learning techniques on seismic waveform data.




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

Installation
============

To ensure the GPU-powered `PyTorch <https://github.com/pytorch/pytorch>`_ ,
first isntall PyTorch using the offical guide:
https://pytorch.org/get-started/locally/ and then install
`Yews <https://github.com/lijunzh/yews>`_ via one of the following approaches:

conda:

.. code:: bash

   conda install -c lijunzhu -c pytorch yews

conda-forge:

.. code:: bash
   conda install -c conda-forge yews


pip:

.. code:: bash

   pip install yews


From source:

.. code:: bash

    python setup.py install


Note:

#. Running the above command without first installing PyTorch may still work.
   Depending on the OS, you may get either the GPU or CPU version of PyTorch.
   For example, MacOS currently will get the CPU Pytorch while Linux will get
   the GPU PyTorch by default. Please refer to
   https://pytorch.org/get-started/locally/ for details.

#. ``Yews`` by itself is ``noarch``, which means it is pure Python and OS
   independent. Most inconsistenciews between OS's are primarily due to the
   upstream difference (e.g. PyTorch, NumPy).

#. ``SciPy`` is an optional dependence, which is installed by default in
   Anaconda; however, ``Yews``'s core functionalities do not depend on
   ``Scipy``. From 0.0.5 and above, ``conda`` and ``pip`` will not
   automatically install ``SciPy``.

#. You can install all ``Yews`` dependencies via ``pip``:

.. code:: bash

   pip install yews[all]




Documentation
=============

You can find the API documentation on the yews website:
https://www.yews.info/docs/

Contributing
============

We appreciate all contributions. If you are planning to contribute back
bug-fixes, please do so without any further discussion. If you plan to
contribute new features, utility functions or extensions, please first open an
issue and discuss the feature with us.
