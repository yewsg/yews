Install PyTorch on Raspberry Pi
===============================

:Platform:  Raspberry Pi 3 B+
:OS:        Raspbian_ Stretch Lite
:Torch:     PyTorch_ 1.1.0
:Python:    Berryconda_

.. contents::

This documentation serves as a memo for installing PyTorch_ (torch + caffe2) on
Raspberry Pi.
Unlike many online tutorials that start with the native python, I choose to
compile it against the Miniconda_ python.
Since the Miniconda is only available up to Python 3.5, Berryconda_ is used
instead for Python 3.6.

Prerequisities
--------------

- Install Raspbian_ Stretch Lite on SD card:

  #. Download the Raspbian_ image: https://www.raspberrypi.org/downloads/raspbian/
  #. Follow the oneline tutorial: https://www.raspberrypi.org/documentation/installation/installing-images/

- Increase swap to at least 2GB:

   .. code-block::

      # edit the swap file setup
      sudo nano /etc/dphys-swapfile
      # change the CONF_SWAPSIZE = 2048
      # restart swap
      sudo /etc/init.d/dphys-swapfile stop
      sudo /etc/init.d/dphys-swapfile start

- Install build tools

  .. code-block::

      sudo apt install git cmake ninja gfortran


Berryconda
----------

#. Download Berryconda_ 3: https://github.com/jjhelmus/berryconda/releases/download/v2.0.0/Berryconda3-2.0.0-Linux-armv7l.sh
#. Install via the ``sh`` script

   .. code-block::

      bash Berryconda3-2.0.0-Linux-armv7l.sh

#. Make ``conda`` available in normal mode:

   .. code-block::

      echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc

#. Install Python dependencies:

   .. code-block::

      conda activate
      conda install cython m4 cmake pyyaml numpy
      conda deactivate

PyTorch
-------

Now we can install PyTorch from source, which contains both ``torch`` and
``caffe2`` packages.

#. Clone the most current repository from GitHub:

   .. code-block::

      cd git
      git clone --recursive https://github.com/pytorch/pytorch
      cd pytorch
      git submodule update --init --recursive

#. Disable component not available on Raspberry Pi by setting the build
   environment variables:

   .. code-block::

      export NO_CUDA=1
      export USE_FBGEMM=0
      export BUILD_TEST=0
      export USE_MIOPEN=0
      export USE_DISTRIBUTED=0
      export USE_MKLDNN=0
      export USE_NNPACK=0
      export USE_QNNPACK=0
      export BUILD_CAFFE2_OPS=0

#. Build from source

   .. code-block::

      conda activate
      python setup.py build
      conda deactivate
Reference
---------

This section shows the useful online tutorial/links I follow.

.. _Raspbian: https://www.raspberrypi.org/downloads/raspbian/
.. _Miniconda: https://repo.continuum.io/miniconda/
.. _Berryconda: https://github.com/jjhelmus/berryconda
.. _PyTorch: https://github.com/pytorch/pytorch
