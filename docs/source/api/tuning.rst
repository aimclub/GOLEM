Tuning of graph parameters
==========================

``SimultaneousTuner`` and ``SequentialTuner`` work with internal graph representation (also called `optimization graph`, see `Adaptation of Graphs`_).
To optimise custom domain graph pass ``adapter``. If your graph class is inherited from ``OptGraph`` no adapter is needed.
Tuners optimise parameters stored in ``OptNode.parameters``.

To specify parameters search space use ``SearchSpace`` class.
Initialize ``SearchSpace`` with dictionary of the form
``{'operation_name': {'param_name': (hyperopt distribution function, [sampling scope]), ...}, ...}``.

.. code::

    import numpy as np
    from hyperopt import hp
    from golem.core.tuning.search_space import SearchSpace


    params_per_operation = {
        'operation_name_1': {
            'parameter_name_1': (hp.uniformint, [2, 7]),
            'parameter_name_2': (hp.loguniform, [np.log(1e-3), np.log(1)])
        },
        'operation_name_2': {
            'parameter_name_1': (hp.choice, [["first", "second", "third"]]),
            'parameter_name_2': (hp.uniform, [0.05, 1.0])
        }}

    search_space = SearchSpace(params_per_operation)

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
