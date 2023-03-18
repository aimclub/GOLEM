Structural Analysis
===============================

The structural analysis is used to identify the influential parts of a graph
(node or edges) and assess the magnitude of the impact of each part.
It is widely used to make data-driven decisions and to test assumptions
made during the analysis. The simulations are used to test the model's robustness
and identify the threshold values beyond which the outcome will have a positive or
negative impact.

Main concepts
-------------

The main logic of Structural Analysis is pretty simple: the algorithm tries to delete
every part of the graph and therefore evaluates the changed one and compares its metric
with the original one's. If metric with applied change is higher than it was before --
than the change should be applied, otherwise not.

There are two pseudocode samples to illustrate the way SA works.

.. image:: img/sa.png

.. image:: img/sa_optimization.png


Usage example
-------------



