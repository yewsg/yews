=========
Changelog
=========

Development versions
====================

Experimental versions
=====================

Version 0.0.5, Under Development
--------------------------------

- experiment with Raspberry Pi deployment
- fix transforms bug during inference
- simplify CPIC model
- enable batch size during deployment
- reorganize dataset module
- create new examples based on functional programming interface
- temporarily disable progress bar during data export

Version 0.0.5, 2019-05-01
--------------------------------

- fix setup.cfg bug (tests_require instead of test_require)
- sync meta.yaml and setup.cfg for consistent build
- documents improvements
- put packaged datasets in a separate module
- wenchuan dataset packaged (link disable temporarily for release grant)
- mariana dataset packaged (link disable temporarily for release grant)
- redo numpy memmap related block
- add export_dataset() method to ``BaseDataset``
- add obspy I/O support w/ tests
- mariana dataset from source w/o tests
- add prgress bar for ``export_dataset()`` method in ``BaseDataset``
- fake ``scipy.special.expit`` to simplify dependency
- partially fake ``tqdm.tdqm`` to simplify dependency

Version 0.0.4, 2019-04-18
-------------------------

- single source installation metadata in setup.cfg
- improve Makefile for automation
- use Scipy as an extra for special transforms
- add pre-commit for automaticall commit checking
- add CHANGELOOG.rst
- add ``is_valid()`` and ``handle_invalid`` to ``yews.datasets.BaseDataset``
- add ``yews.datasets.utils`` to handle urls and bz2 files.
- add ``yews.datasets.wenchuan``
- add ``memory limit`` to ``yews.datasets`` to determine when to load the
  entire dataset into memory.

Version 0.0.3, 2019-04-18
-------------------------

- imporove documentation
- move documentation to a separate repo for Netlify automatic deployment
- add Makefile for automatic release

Version 0.0.2, 2019-04-17
-------------------------

- add logo
- revamp ``yews.transforms`` with unittest
- add sphinx docs
- add CI tools: travis-ci
- add CI tools: appveyor
- add CI tools: codecov
- add release in conda-forge
- add release in anaconda cloud
- add ``yews.transforms.ToInt``

Version 0.0.1, 2019-04-10
-------------------------

- first experimental release
- add ``yews.datasets``
- add ``yews.transforms``
- add ``yews.train``
- add ``yews.models`` for CPIC models
- add an example for Wenchuan aftershock dataset
- add release in pypi
