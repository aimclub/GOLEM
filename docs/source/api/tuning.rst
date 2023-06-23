Tuning of graph parameters
==========================

All tuners work with internal graph representation (also called `optimization graph`, see `Adaptation of Graphs`_).
To optimise custom domain graph pass ``adapter``. If your graph class is inherited from ``OptGraph`` no adapter is needed.
Tuners optimise parameters stored in ``OptNode.parameters``.

Multi-objective optimisation is supported only by ``OptunaTuner``.

To specify parameters search space use ``SearchSpace`` class.
Initialize ``SearchSpace`` with dictionary of the form
``{'operation_name': {'param_name': { 'hyperopt-dist': <hyperopt distribution function>,
'sampling-scope': [sampling scope], 'type': <type of parameter>}, ...}, ...}``.
Three types of parameters are available: `continuous`, `discrete` and `categorical`.

.. code::

    import numpy as np
    from hyperopt import hp
    from golem.core.tuning.search_space import SearchSpace


    params_per_operation = {
        'operation_name_1': {
            'parameter_name_1': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'parameter_name_2': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'}
        },
        'operation_name_2': {
            'parameter_name_1': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["first", "second", "third"]],
                'type': 'categorical'},
            'parameter_name_2':
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'}
        }}

    search_space = SearchSpace(params_per_operation)

Simultaneous
~~~~~~~~~~~~

You can tune all parameters of graph nodes simultaneously using ``SimultaneousTuner``, ``OptunaTuner`` or ``IOptTuner``.

.. note::
   ``IOptTuner`` implements deterministic algorithm.

   For now ``IOptTuner`` can not be constrained by time, so constrain execution by number of iterations.

   Also ``IOptTuner`` can optimise only `continuous` and `discrete` parameters but not `categorical` ones.
   `Categorical` parameters will be ignored while tuning.

   ``IOptTuner`` is implemented using `IOpt library`_. See the `documentation`_ (in Russian) to learn more about
   the optimisation algorithm.

.. automodule:: golem.core.tuning.simultaneous
   :members:

.. autoclass:: golem.core.tuning.iopt_tuner.IOptTuner
   :members:

Sequential
~~~~~~~~~~

``SequentialTuner`` allows you to tune graph parameters sequentially node by node.

.. automodule:: golem.core.tuning.sequential
   :members:

.. _Adaptation of Graphs: https://thegolem.readthedocs.io/en/latest/advanced/adaptation.html
.. _IOpt library: https://github.com/aimclub/iOpt
.. _documentation: https://iopt.readthedocs.io/ru/latest/introduction.html