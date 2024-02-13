.. image:: /docs/source/img/golem_logo-02.png
   :alt: Logo of GOLEM framework
   :align: center
   :width: 500

.. class:: center

    |sai| |itmo|

    |python| |pypi| |build| |integration| |coverage| |docs| |license| |tg| |rus| |mirror|


Graph Optimization and Learning by Evolutionary Methods
-------------------------------------------------------

GOLEM is an open-source AI framework for optimization and learning of structured graph-based models with meta-heuristic
methods. It is centered around 2 ideas:

1. The potential of meta-heuristic methods in complex problem spaces.

The focus on meta-heuristics allows approaching the kinds of problems where gradient-based learning methods (notably, neural networks)
can't be easily applied, like optimization problems with multiple conflicting objectives or having a combinatorial nature.

2. The importance of structured models in multiple problem domains.

Graph-based learning enables solutions in the form of structured and hybrid probabilistic models, not to mention
that a wide range of domain-specific problems have a natural formulation in the form of graphs.

Together this constitutes an approach to AI that potentially leads to structured, intuitive, interpretable methods and
solutions for a wide range of tasks.


Core Features
=============

- **Structured** models with joint optimization of graph structure and properties (node attributes).
- **Metaheuristic** methods (mainly evolutionary) applicable to any task with a well-defined objective.
- **Multi-objective** optimization that can take into account both quality and complexity.
- **Constrained** optimization with support for arbitrary domain-specific constraints.
- **Extensible** to new domains.
- **Interpretable** thanks to meta-heuristics, structured models, and visualisation tools.
- **Reproducible** thanks to rich optimization history and model serialization.


Applications
============

GOLEM is potentially applicable to any optimization problem structures:

- that can be represented as directed graphs;
- that have some clearly defined fitness function on them.

Graph models can represent fixed structures (e.g. physical models such as truss structures) or functional models that
define a data-flow or inference process (e.g. bayesian networks that can be fitted and queried).

Examples of GOLEM applications:

- Automatic Machine Learning (AutoML) with optimal ML pipelines search in `FEDOT framework <https://github.com/aimclub/FEDOT>`_
- Bayesian network structure search in `BAMT framework <https://github.com/aimclub/BAMT>`_
- Differential equation discovery for physical models in `EPDE framework <https://github.com/ITMO-NSS-team/EPDE>`_
- Geometric design of physical objects in `GEFEST framework <https://github.com/aimclub/GEFEST>`_
- `Neural architecture search <https://github.com/ITMO-NSS-team/nas-fedot>`_

As GOLEM is a general-purpose framework, it's easy to imagine potential applications, for example, finite state automata search
for robotics control or molecular graph learning for drug discovery, and more.


Installation
============

GOLEM can be installed with ``pip``:

.. code-block::

  $ pip install thegolem


Quick Start Example
===================

Following example demonstrates graph search using reference graph & edit distance metric. Optimizer is set up with a minimal set of parameters and simple single-point mutations. For more details see examples `simple_run.py <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/simple_run.py>`_, `graph_search.py <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/graph_search.py>`_ and `tree_search.py <https://github.com/aimclub/GOLEM/blob/main/examples/synthetic_graph_evolution/tree_search.py>`_ in directory `examples/synthetic_graph_evolution <https://github.com/aimclub/GOLEM/tree/main/examples/synthetic_graph_evolution>`_.

.. code-block:: python

    def run_graph_search(size=16, timeout=8):
        # Generate target graph sought by optimizer using edit distance objective
        node_types = ('a', 'b')  # Available node types that can appear in graphs
        target_graph = generate_labeled_graph('tree', size, node_types)
        objective = Objective(partial(tree_edit_dist, target_graph))
        initial_population = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]

        # Setup optimization parameters
        requirements = GraphRequirements(timeout=timedelta(minutes=timeout))
        gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter(), available_node_types=node_types)
        algo_params = GPAlgorithmParameters(pop_size=30)

        # Build and run the optimizer
        optimiser = EvoGraphOptimizer(objective, initial_population, requirements, gen_params, algo_params)
        found_graphs = optimiser.optimise(objective)

        # Visualize results
        found_graph = gen_params.adapter.restore(found_graphs[0])  # Transform back to NetworkX graph
        draw_graphs_subplots(target_graph, found_graph, titles=['Target Graph', 'Found Graph'])
        optimiser.history.show.fitness_line()
        return found_graph


Tracing the lineage of the found_graph reveals how genetic operators (mutations, crossovers, etc.) are applied to a random graph one after another, eventually leading to the target graph:

.. image:: /docs/source/img/evolution_process.gif
   :alt: Evolution process
   :align: center

One can also notice that despite the fact that the edit distance generally decreases along the genealogical path, the optimizer sometimes sacrifices local fitness gain of some graphs in order to achieve diversity and thus obtain the best possible solution at the end.

Project Structure
=================

The repository includes the following packages and directories:

- Package ``core`` contains the main classes and scripts.
- Package ``core.adapter`` is responsible for transformation between domain graphs and internal graph representation used by optimisers.
- Package ``core.dag`` contains classes and algorithms for representation and processing of graphs.
- Package ``core.optimisers`` contains graph optimisers and all related classes (like those representing fitness, individuals, populations, etc.), including optimization history.
- Package ``core.optimisers.genetic`` contains genetic (also called evolutionary) graph optimiser and operators (mutation, selection, and so on).
- Package ``core.utilities`` contains utilities and data structures used by other modules.
- Package ``serializers`` contains class ``Serializer`` with required facilities, and is responsible for serialization of project classes (graphs, optimization history, and everything related).
- Package ``visualisation`` contains classes that allow to visualise optimization history, graphs, and certain plots useful for analysis.
- Package ``examples`` includes several use-cases where you can start to discover how the framework works.
- All unit and integration tests are contained in the ``test`` directory.
- The sources of the documentation are in the ``docs`` directory.


Current R&D and future plans
============================

Any contribution is welcome. Our R&D team is open for cooperation with other scientific teams as well as with industrial partners.

Contribution Guide
==================

- The contribution guide is available in the `repository </docs/source/contribution.rst>`__.

Acknowledgments
===============

We acknowledge the contributors for their important impact and the participants of the numerous scientific conferences and
workshops for their valuable advice and suggestions.

Supported by
============

The study is supported by the Research `Center Strong Artificial Intelligence in Industry <https://sai.itmo.ru/>`_
of `ITMO University <https://itmo.ru/>`_ as part of the plan of the center's program: 
Development and testing of an experimental prototype of the library of strong AI algorithms 
in terms of basic algorithms of automatic ML for structural training of composite AI models, 
including automation of feature selection

Contacts
========
- `Telegram channel <https://t.me/FEDOT_helpdesk>`_ for solving problems and answering questions about FEDOT
- `Natural System Simulation Team <https://itmo-nss-team.github.io/>`_
- `Nikolay Nikitin <https://scholar.google.com/citations?user=eQBTGccAAAAJ&hl=ru>`_, AutoML Lead (nnikitin@itmo.ru)
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

Papers that describe applications of GOLEM:
===========================================

There are various cases solved with GOLEM's algorithms:

- Algorithms for time series forecasting pipeline design: Sarafanov M., Pokrovskii V., Nikitin N. O. Evolutionary Automated Machine Learning for Multi-Scale Decomposition and Forecasting of Sensor Time Series //2022 IEEE Congress on Evolutionary Computation (CEC). – IEEE, 2022. – С. 01-08.

- Algorithms for acoustic equation discovery: Hvatov A. Data-Driven Approach for the Floquet Propagator Inverse Problem Solution //ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). – IEEE, 2022. – С. 3813-3817.

- Algorithms for PDE discovery: Maslyaev M., Hvatov A. Solver-Based Fitness Function for the Data-Driven Evolutionary Discovery of Partial Differential Equations //2022 IEEE Congress on Evolutionary Computation (CEC). – IEEE, 2022. – С. 1-8.

- Algorithms for structural learning of Bayesian Networks: Deeva I., Kalyuzhnaya A. V., Alexander V. Boukhanovsky Adaptive Learning Algorithm for Bayesian Networks Based on Kernel Mixtures Distributions//International Journal of Artificial Intelligence. – 2023. - Т.21. - №. 1. - С. 90.


.. |docs| image:: https://readthedocs.org/projects/thegolem/badge/?version=latest
    :target: https://thegolem.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |build| image:: https://github.com/aimclub/GOLEM/actions/workflows/unit-build.yml/badge.svg?branch=main
   :alt: Build Status
   :target: https://github.com/aimclub/GOLEM/actions/workflows/unit-build.yml

.. |integration| image:: https://github.com/aimclub/GOLEM/actions/workflows/integration-build.yml/badge.svg?branch=main
   :alt: Integration Build Status
   :target: https://github.com/aimclub/GOLEM/actions/workflows/integration-build.yml

.. |coverage| image:: https://codecov.io/gh/aimclub/GOLEM/branch/main/graph/badge.svg
   :alt: Coverage Status
   :target: https://codecov.io/gh/aimclub/GOLEM

.. |pypi| image:: https://img.shields.io/pypi/v/thegolem.svg
   :alt: PyPI Package Version
   :target: https://img.shields.io/pypi/v/thegolem

.. |python| image:: https://img.shields.io/pypi/pyversions/thegolem.svg
   :alt: Supported Python Versions
   :target: https://img.shields.io/pypi/pyversions/thegolem

.. |license| image:: https://img.shields.io/github/license/aimclub/GOLEM
   :alt: Supported Python Versions
   :target: https://github.com/aimclub/GOLEM/blob/main/LICENSE.md

.. |downloads_stats| image:: https://static.pepy.tech/personalized-badge/thegolem?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
   :target: https://pepy.tech/project/thegolem

.. |tg| image:: https://img.shields.io/badge/Telegram-Group-blue.svg
   :alt: Telegram Chat
   :target: https://t.me/FEDOT_helpdesk

.. |by-golem| image:: http://img.shields.io/badge/powered%20by-GOLEM-orange.svg?style=flat
   :target: http://github.com/aimclub/GOLEM
   :alt: Powered by GOLEM

.. |rus| image:: https://img.shields.io/badge/lang-ru-yellow.svg
            :target: /README.rst

.. |ITMO| image:: https://github.com/aimclub/open-source-ops/blob/add_badge/badges/ITMO_badge.svg
   :alt: Acknowledgement to ITMO
   :target: https://en.itmo.ru/en/

.. |SAI| image:: https://github.com/aimclub/open-source-ops/blob/add_badge/badges/SAI_badge.svg
   :alt: Acknowledgement to SAI
   :target: https://sai.itmo.ru/

.. |mirror| image:: https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162
   :alt: GitLab mirror for this repository
   :target: https://gitlab.actcognitive.org/itmo-nss-team/GOLEM