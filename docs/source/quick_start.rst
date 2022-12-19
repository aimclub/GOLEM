Quickstart
==========

GOLEM Framework quick start guide

Installation from GitHub
------------------------
  | git clone https://github.com/aimclub/GOLEM.git
  | cd GOLEM
  | pip install -r requirements.txt
  | pytest -s test/

How to install with pip
-----------------------
.. code::

 pip install https://github.com/aimclub/GOLEM/archive/main.zip


Task definition
------------------

The core notions of the framework are *Graph*, *Optimizer*, *Objective* and *Fitness*. GOLEM runs the optimizer, which searches for the graphs that have better fitness by the given objective.

The usual workflow for solving the task with GOLEM includes these steps:

1. Define the available nodes from which the graph will be built.
2. Import the optimizer and define the settings for it.
3. Define the objective, that is a function which for a given graph computes fitness score.
4. Run the optimizer with given objective and get the best graphs.

For a code example you can look into the `simple example <https://github.com/aimclub/GOLEM/blob/main/examples/graph_model_optimization.py>`_.

Adapter for domain structures
-----------------------------

Commonly your domain task works with specific structures. These can be some special graph models, 3D models, etc. Before running GOLEM for such tasks you need to define the *graph adapter* that will transform your domain graphs to universal graph representation, that GOLEM uses internally.
For this you either need to subclass the :py:class:`golem.core.adapter.adapter.BaseOptimizationAdapter` or use one of the existing adapters. Currently there's one non-trivial adapter for `NetworkX <https://networkx.org/>`_ directed graphs. You can read more about it in :doc:`this document </advanced/nx-interop>`_.
You can also find example using this adapter in this example on :doc:`synthetic graph search </examples/abstract>`_.
