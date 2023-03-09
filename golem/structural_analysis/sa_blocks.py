import datetime
import os

from typing import List, Callable, Any

from golem.core.log import LoggerAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.timer import OptimisationTimer
from golem.structural_analysis.graph_sa.sa_approaches_repository import SUBTREE_DELETION, NODE_DELETION, \
    NODE_REPLACEMENT, EDGE_DELETION
from golem.structural_analysis.graph_sa.sa_based_postproc import sa_postproc


def structural_analysis_version_1(graphs_to_handle: List[OptGraph],
                                   timeout: datetime.timedelta,
                                   task_type: Any,
                                   log: LoggerAdapter, n_jobs: int,
                                   objective: Callable = None,
                                   save_path: str = None) -> List[OptGraph]:
    """ In this version of SA approaches are applied one after another """
    if not isinstance(objective, list):
        objective = [objective]
    result_graphs = []
    try:
        with OptimisationTimer(timeout=timeout) as t:
            for idx, graph in enumerate(graphs_to_handle):
                log.message(f'Length of current graph with idx {idx}: {len(graph.nodes)}')
                for approach in [SUBTREE_DELETION,
                                 NODE_DELETION, NODE_REPLACEMENT,
                                 EDGE_DELETION]:

                    graph = sa_postproc(approaches_names=[approach], graph_before_sa=graph, timer=t,
                                           task_type=task_type,
                                           log=log, n_jobs=n_jobs, objectives=objective,
                                           save_path=save_path)
                result_graphs.append(graph)
            log.message(f'Time spent for SA: {t.seconds_from_start}')

    except Exception as ex:
        log.error(f'Exception occurred in SA block version 1: {ex}')
        return graphs_to_handle
    return result_graphs


def structural_analysis_version_2(graphs_to_handle: List[OptGraph],
                                   timeout: datetime.timedelta,
                                   task_type: Any,
                                   log: LoggerAdapter, n_jobs: int,
                                   objective: Callable = None,
                                   save_path: str = None) -> List[OptGraph]:
    """ In this version of SA approaches are calculated on every iteration all together and
    then only approach with the best metric is applied """
    if not isinstance(objective, list):
        objective = [objective]
    result_graphs = []
    try:
        with OptimisationTimer(timeout=timeout) as t:
            for idx, graph in enumerate(graphs_to_handle):
                log.message(f'Length of current graph with idx {idx}: {len(graph.nodes)}')
                res_graph = sa_postproc(approaches_names=[SUBTREE_DELETION,
                                                             NODE_DELETION, NODE_REPLACEMENT,
                                                             EDGE_DELETION
                                                             ],
                                           graph_before_sa=graph, timer=t,
                                           task_type=task_type,
                                           log=log, n_jobs=n_jobs, objectives=objective,
                                           save_path=save_path)
                result_graphs.append(res_graph)
            log.message(f'Time spent for SA: {t.seconds_from_start}')

    except Exception as ex:
        log.error(f'Exception occurred in SA block version 2: {ex}')
        return graphs_to_handle
    return result_graphs
