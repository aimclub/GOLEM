Tuning of graph parameters
==========================

``SimultaneousTuner`` and ``SequentialTuner`` work with internal graph representation (also called `optimization graph`, see `Adaptation of Graphs`_).
To optimise custom domain graph pass ``adapter``. If your graph class is inherited from ``OptGraph`` no adapter is needed.
Tuners optimise parameters stored in ``OptNode.parameters``.

To specify parameters search space use ``SearchSpace`` class.

Simultaneous
~~~~~~~~~~~~

You can tune all parameters of graph nodes simultaneously using ``SimultaneousTuner``.

.. automodule:: golem.core.tuning.simultaneous
   :members:

Sequential
~~~~~~~~~~

``SequentialTuner`` allows you to tune graph parameters sequentially node by node.

.. automodule:: golem.core.tuning.sequential
   :members:

.. _Adaptation of Graphs: https://thegolem.readthedocs.io/en/latest/advanced/adaptation.html
