import json
import multiprocessing
import os
from copy import deepcopy
from typing import List, Dict, Union, Any, Tuple, Optional, Callable

from golem.core.log import LoggerAdapter, default_log
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import project_root
from golem.structural_analysis.graph_labels_using_sa import draw_nx_dag
from golem.structural_analysis.pipeline_sa.edge_sa_approaches import EdgeAnalyzeApproach
from golem.structural_analysis.pipeline_sa.node_sa_approaches import NodeAnalyzeApproach
from golem.structural_analysis.pipeline_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.pipeline_sa.sa_approaches_repository import StructuralAnalysisApproachesRepository
from golem.structural_analysis.pipeline_sa.sa_requirements import StructuralAnalysisRequirements


def sa_postproc(approaches_names: List[str], pipeline_before_sa: OptGraph,
                task_type: Any, data: Any = None, objectives: List[Callable] = None,
                metrics: List[str] = None,
                timer: OptimisationTimer = None,
                log: Optional[LoggerAdapter] = None, max_iter: int = 20, n_jobs: int = -1,
                is_visualize: bool = False, is_save_results_to_json: bool = False, is_preproc: bool = True,
                **kwargs) -> Union[OptGraph, Tuple[OptGraph, Dict[str, int]]]:
    """
    Function for pipeline post-processing using structural analysis

    :param approaches_names: names of SA approaches to apply to the pipeline.
    Approaches can be applied both iteratively and all at once. In the first case,
    the function must be called separately for each approach, and in the second - it must be called once,
    specifying all the necessary approaches
    :param pipeline_before_sa: OptGraph that will be analyzed
    :param task_type: type of solving task
    :param data: input data
    :param objectives: list of objective functions for computing metrics
    :param metrics: metrics to use for optimization.
    All metrics will be calculated but result will be made using the first one.
    :param timer: timer to check if the time allotted for structural analysis has expired
    :param log: log
    :param max_iter: max possible iteration. It is desirable to set this parameter,
    since this approach can work for a very long time
    :param n_jobs: the number of jobs to run in parallel
    :param is_visualize: a boolean flag indicating whether to visualize results or not
    :param is_save_results_to_json: a boolean flag indicating whether to save results to json or not
    :param is_preproc: is preprocessing needed or not
    :return: optimized pipeline
    """
    if not log:
        log = default_log(prefix=__name__)

    if not objectives:
        objectives = _get_objectives_from_data(data=data, metrics=metrics, task_type=task_type)
    approaches_repo = StructuralAnalysisApproachesRepository()
    approaches = [approaches_repo.approach_by_name(approach_name) for approach_name in approaches_names]

    # what actions were applied on the pipeline and how many
    actions_applied = dict.fromkeys(approaches_names, 0)

    save_path = kwargs.get('save_path')
    if save_path:
        kwargs.pop('save_path')

    new_pipeline = deepcopy(pipeline_before_sa)
    analysis_results = _analyse(pipeline=new_pipeline, objectives=objectives, task_type=task_type,
                                approaches=approaches,
                                is_visualize=is_visualize, is_save_results_to_json=is_save_results_to_json,
                                is_preproc=is_preproc, timer=timer, n_jobs=n_jobs, **kwargs)
    converged = False
    iter = 0

    if len(list(analysis_results.values())) == 0 or not analysis_results.keys():
        log.message(f'{iter} actions were taken during SA')
        log.message(f'The following actions were applied during SA: {actions_applied}')
        return pipeline_before_sa

    while not converged:
        iter += 1
        worst_approach_name = None
        worst_approach_result = 0
        entity_to_change = None
        for section in list(analysis_results.values()):
            for entity in section.keys():
                for approach_key in section[entity].keys():
                    if section[entity][approach_key]['loss'][0] > worst_approach_result:
                        worst_approach_result = section[entity][approach_key]['loss'][0]
                        worst_approach_name = approach_key
                        entity_to_change = entity
        if save_path:
            _save_iteration_results_to_json(analysis_results=analysis_results,
                                            save_path=save_path)
        if worst_approach_result > 1.0:
            postproc_method = approaches_repo.postproc_method_by_name(worst_approach_name)
            new_pipeline = postproc_method(results=analysis_results, pipeline=new_pipeline, entity=entity_to_change)
            actions_applied[f'{worst_approach_name}'] += 1
            if timer is not None and timer.is_time_limit_reached():
                break

            if max_iter and iter >= max_iter:
                break

            analysis_results = _analyse(pipeline=new_pipeline, objectives=objectives, task_type=task_type,
                                        approaches=approaches,
                                        is_visualize=is_visualize, is_save_results_to_json=is_save_results_to_json,
                                        is_preproc=is_preproc, timer=timer, n_jobs=n_jobs, **kwargs)
        else:
            converged = True

    if save_path:
        _save_iteration_results(pipeline_before_sa=pipeline_before_sa, save_path=save_path)

    log.message(f'{iter} iterations passed during SA')
    log.message(f'The following actions were applied during SA: {actions_applied}')
    if isinstance(new_pipeline, OptGraph):
        return new_pipeline
    else:
        return pipeline_before_sa


def _analyse(pipeline: OptGraph, objectives: List[Callable],
             task_type: Any,
             approaches: List[Union[NodeAnalyzeApproach, EdgeAnalyzeApproach]],
             is_visualize: bool = False,
             is_save_results_to_json: bool = False,
             is_preproc: bool = True,
             number_of_replace_operations_nodes: int = 5,
             number_of_replace_operations_edges: int = 5,
             n_jobs: int = -1,
             timer: OptimisationTimer = None,
             ) -> Dict[str, Any]:

    sa_requirements = \
        StructuralAnalysisRequirements(replacement_number_of_random_operations_nodes=
                                        number_of_replace_operations_nodes,
                                       replacement_number_of_random_operations_edges=
                                        number_of_replace_operations_edges,
                                       is_visualize=is_visualize,
                                       is_save_results_to_json=is_save_results_to_json)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    analysis_results = GraphStructuralAnalysis(pipeline=pipeline, objectives=objectives,
                                                task_type=task_type,
                                                is_preproc=is_preproc,
                                                approaches=approaches,
                                                requirements=sa_requirements).analyze(n_jobs=n_jobs,
                                                                                         timer=timer)
    return analysis_results


def get_objective_evaluate(metric: str, data: Any,
                           cv_folds: Optional[int] = None, validation_blocks: Optional[int] = None):
    return


def _get_objectives_from_data(data: Any, metrics: List[str], task_type: Any):
    """ Makes evaluating objective function using metrics and data """
    return


def _save_iteration_results(pipeline_before_sa: OptGraph, save_path: str = None):
    """ Save visualizations for SA per iteration """
    json_path = os.path.join(save_path, 'results_per_iteration.json')
    pipeline_save_path = os.path.join(save_path, 'result_pipelines')
    pipeline_before_sa.save(pipeline_save_path)
    if not os.path.exists(pipeline_save_path):
        os.makedirs(pipeline_save_path)
    try:
        draw_nx_dag(graph=pipeline_before_sa, save_path=save_path, json_path=json_path)
    except Exception as ex:
        log = default_log('draw_viz')
        log.error(f'Visualisation failed: {ex}')


def _save_iteration_results_to_json(analysis_results: dict, save_path: str = None):
    """ Save SA actions scores in json file """
    if save_path:
        save_path = os.path.join(save_path, 'results_per_iteration.json')
    else:
        save_path = os.path.join(project_root(), 'examples', 'structural_analysis',
                                 'show_sa_on_graph', 'results_per_iteration.json')
    if not os.path.exists(save_path):
        json_data = [analysis_results]
        with open(save_path, 'w') as file:
            file.write(json.dumps(json_data, indent=2, ensure_ascii=False))
    else:
        data = json.load(open(save_path))
        data.append(analysis_results)
        with open(save_path, 'w', encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
