.. image:: docs/golem_logo.png
   :alt: Logo of GOLEM framework

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi| |py_7| |py_8| |py_9|
   * - tests
     - | |build|
   * - docs
     - |docs|
   * - license
     - | |license|
   * - support
     - | |tg|


.. end-badges

**GOLEM**: Graph Optimization and Learning by Evolutionary Methods

GOLEM is an open-source AI framework for optimization and learning of structured graph-based models with meta-heuristic methods.

GOLEM is centered around 2 ideas:

1. Potential of meta-heuristic methods in complex problem spaces.
Focus on meta-heuristics allows approaching kinds of problems where gradient-based learning methods (notably, neural networks) can't be easily applied, like optimization problems with multiple conflicting objectives or having combinatorial character.

2. Importance of structured models in many problem domains.
Graph-based learning enables solutions in the form of structured and hybrid probabilistic models, not to mention that a wide range of domain-specific problems have a natural formulation in the form of graphs.

Together this constitutes an approach to AI that potentially leads to structured, intuitive, interpretable methods and solutions for a wide range of tasks.


Core Features
=============

- **Structured** models with joint optimization of graph structure (connections) and properties (node attributes) for directed graphs.
- **Metaheuristic** methods (notably, evolutionary graph optimizer), applicable for any task with defined objective.
- **Multi-objective** optimization that can take into account both quality & complexity metrics.
- **Constrained** optimization with support for arbitrary domain-specific constraints.
- **Extensible** to new problem domains with a flexible adaptation subsystem that transforms domain structures into optimized graphs.
- **Interpretable** thanks to the use of meta-heuristics and structured models together with visualisation facilities.
- **Reproducible** thanks to rich optimization history and model serialization.


GOLEM Applications
==================

GOLEM is potentially applicable for any optimization problem structures
- that can be represented as directed graphs;
- with some clearly defined fitness function on them.

Graph models can represent fixed structures (e.g. physical model such as truss structures) or functional models that define a data-flow or inference process (e.g. bayesian networks that can be fitted and queried).

Examples of GOLEM applications:

- Automatic Machine Learning (AutoML) with optimal ML pipelines search in `FEDOT framework <https://github.com/aimclub/FEDOT>`_
- Bayesian network structure search in `BAMT framework <https://github.com//FEDOT>`_
- Differential equation discovery for physical models in `EPDE framework <https://github.com/ITMO-NSS-team/EPDE>`_
- Geometric design of physical objects in `GEFEST framework <https://github.com/aimclub/GEFEST>`_
- `Neural architecture search <https://github.com/ITMO-NSS-team/nas-fedot>`_

As GOLEM is a general-purpose, it's easy to imagine fore potential applications, for example, finite state automata search for robotics control or molecular graph learning for drug discovery, and more.

..
    TODO:
    Installation
    ============

    GOLEM can be installed with ``pip``:

    .. code-block::

      $ pip install golem



Current R&D and future plans
============================

Any contribution is welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.

Contribution Guide
==================

- The contribution guide is available in the `repository <https://github.com/nccr-itmo/FEDOT/blob/master/docs/source/contribution.rst>`__.

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and workshops for their valuable advice and suggestions.

Supported by
============

The project is maintained by the research team of the Natural Systems Simulation Lab. It is a part of the `National Center for Cognitive Research of ITMO University <https://actcognitive.org/>`_, that supports research and development of the project.

Contacts
========
- `Telegram channel for solving problems and answering questions on FEDOT <https://t.me/FEDOT_helpdesk>`_
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Anna Kalyuzhnaya <https://scholar.google.com/citations?user=bjiILqcAAAAJ&hl=ru>`_, Team leader (anna.kalyuzhnaya@itmo.ru)
- `Newsfeed <https://t.me/NSS_group>`_
- `Youtube channel <https://www.youtube.com/channel/UC4K9QWaEUpT_p3R4FeDp5jA>`_

Citation
========

If you use our project in your work or research, we would appreciate citations.

@article{nikitin2021automated,
  title = {Automated evolutionary approach for the design of composite machine learning pipelines},
  author = {Nikolay O. Nikitin and Pavel Vychuzhanin and Mikhail Sarafanov and Iana S. Polonskaia and Ilia Revin and Irina V. Barabanova and Gleb Maximov and Anna V. Kalyuzhnaya and Alexander Boukhanovsky},
  journal = {Future Generation Computer Systems},
  year = {2021},
  issn = {0167-739X},
  doi = {https://doi.org/10.1016/j.future.2021.08.022}}

@inproceedings{polonskaia2021multi,
  title={Multi-Objective Evolutionary Design of Composite Data-Driven Models},
  author={Polonskaia, Iana S. and Nikitin, Nikolay O. and Revin, Ilia and Vychuzhanin, Pavel and Kalyuzhnaya, Anna V.},
  booktitle={2021 IEEE Congress on Evolutionary Computation (CEC)},
  year={2021},
  pages={926-933},
  doi={10.1109/CEC45853.2021.9504773}}


Other papers - in `ResearchGate <https://www.researchgate.net/project/Evolutionary-multi-modal-AutoML-with-FEDOT-framework>`_.

.. |docs| image:: https://readthedocs.org/projects/ebonite/badge/?style=flat
   :target: https://fedot.readthedocs.io/en/latest/
   :alt: Documentation Status

.. |build| image:: https://github.com/nccr-itmo/FEDOT/workflows/Build/badge.svg?branch=master
   :alt: Build Status
   :target: https://github.com/nccr-itmo/FEDOT/actions

.. |coverage| image:: https://codecov.io/gh/nccr-itmo/FEDOT/branch/master/graph/badge.svg
   :alt: Coverage Status
   :target: https://codecov.io/gh/nccr-itmo/FEDOT

.. |pypi| image:: https://badge.fury.io/py/fedot.svg
   :alt: Supported Python Versions
   :target: https://badge.fury.io/py/fedot

.. |py_7| image:: https://img.shields.io/badge/python_3.7-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.7-passing-success

.. |py_8| image:: https://img.shields.io/badge/python_3.8-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.8-passing-success

.. |py_9| image:: https://img.shields.io/badge/python_3.9-passing-success
   :alt: Supported Python Versions
   :target: https://img.shields.io/badge/python_3.9-passing-success

.. |license| image:: https://img.shields.io/github/license/nccr-itmo/FEDOT
   :alt: Supported Python Versions
   :target: https://github.com/nccr-itmo/FEDOT/blob/master/LICENSE.md

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/fedot?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/fedot

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
          :target: https://t.me/FEDOT_helpdesk
          :alt: Telegram Chat
