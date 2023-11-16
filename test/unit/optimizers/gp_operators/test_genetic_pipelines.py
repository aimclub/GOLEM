import random
from collections import Counter
from math import ceil
from itertools import product
from typing import Optional

import pytest
from examples.synthetic_graph_evolution.generators import generate_labeled_graph

from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.operators.node import GeneticOperatorTask, TaskStagesEnum, GeneticNode
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, Operator, PopulationT
from golem.core.optimisers.opt_history_objects.individual import Individual


class UncorrectIndividualsCount(Exception):
    pass


class Mock:
    def __init__(self, success_prob: float = 1.0):
        self.success_prob = success_prob

    def __call__(self):
        if random.random() > self.success_prob:
            raise Exception()


class MockOperator(Mock, Operator):
    def __init__(self, *args,
                 individuals_input_count: Optional[int] = None,
                 individuals_output_count: Optional[int] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.individuals_input_count = individuals_input_count
        self.individuals_output_count = individuals_output_count

    def __call__(self, individuals, operation_type = None):
        if ((self.individuals_input_count is not None and len(individuals) > self.individuals_input_count) or
             len(individuals) == 0):
            raise UncorrectIndividualsCount()
        super().__call__()
        if self.individuals_output_count is None:
            return individuals, operation_type
        else:
            return individuals[:1] * self.individuals_output_count, operation_type


class MockEvaluator(Mock, EvaluationOperator):
    def __call__(self, pop):
        super().__call__()
        n_valid = int(ceil(self.success_prob * len(pop)))
        evaluated = random.sample(pop, n_valid)
        return evaluated

def get_rand_population(pop_size: int = 10) -> PopulationT:
    graph_sizes = list(range(5, 15))
    random_pop = [generate_labeled_graph('tree', size=random.choice(graph_sizes),
                                         directed=True)
                  for _ in range(pop_size)]
    graph_pop = BaseNetworkxAdapter().adapt(random_pop)
    individuals = [Individual(graph) for graph in graph_pop]
    return individuals


def get_random_task(pop_size: int = 10, operator_type: str = 'test_operator_type', **params):
    return GeneticOperatorTask(individuals=get_rand_population(pop_size),
                               operator_type=operator_type,
                               **params)


def test_genetic_task_constructor():
    individuals = get_rand_population()
    operator_type = 'test_operator_type'

    task = GeneticOperatorTask(individuals=individuals,
                               operator_type=operator_type)

    # check task constructor
    assert task.individuals == individuals
    assert task.operator_type == operator_type
    assert task.left_tries == 1
    assert task.exception is None
    assert task.stage is TaskStagesEnum.INIT


def test_genetic_failed_task():
    task = get_random_task()

    left_tries = task.left_tries
    stage = task.stage

    exception = Exception('test')
    new_task = task.create_failed_task(exception)

    assert id(new_task) != id(task)
    assert task.exception is None
    assert task.left_tries == left_tries
    assert task.stage == stage

    assert new_task.exception == exception
    assert (task.left_tries - new_task.left_tries) == 1
    assert new_task.stage is TaskStagesEnum.FAIL

    for attr in ('individuals', 'operator_type', 'next_stage_node', 'prev_stage_node', 'parent_task'):
        assert getattr(new_task, attr) == getattr(task, attr)


def test_genetic_successive_task():
    task = get_random_task()

    stage = task.stage
    individuals = task.individuals
    new_individuals = get_rand_population(5)
    new_task = task.create_successive_task(new_individuals)

    assert id(new_task) != id(task)
    assert task.stage == stage
    assert task.individuals == individuals

    assert task.left_tries == new_task.left_tries
    assert new_task.stage is TaskStagesEnum.SUCCESS
    assert new_task.individuals == new_individuals
    assert new_task.parent_task == task

    for attr in ('next_stage_node', 'prev_stage_node'):
        assert getattr(new_task, attr) == getattr(task, attr)


@pytest.mark.parametrize(['stage', 'success_outputs', 'left_tries',
                          'individuals_input_count', 'individuals_output_count',
                          'repeat_count', 'tries_count'],
                         product([TaskStagesEnum.INIT, TaskStagesEnum.SUCCESS],  # stage
                                 [[None], ['1', '2', '3']],  # success_outputs
                                 [1, 3],  # left_tries
                                 [1, 3, None],  # individuals_input_count
                                 [1, 3, None],  # individuals_output_count
                                 [1, 3],  # repeat_count
                                 [1, 3],  # tries_count
                                 ))
def test_genetic_node_with_nonfailed_task(stage, success_outputs, left_tries, individuals_input_count,
                                          individuals_output_count, repeat_count, tries_count):
    pop_size = 10
    node_name = 'test'

    task = get_random_task(pop_size=pop_size, stage=stage, left_tries=left_tries)
    operator = MockOperator(success_prob=1, individuals_input_count=individuals_input_count,
                            individuals_output_count=individuals_output_count)
    node = GeneticNode(name=node_name, operator=operator, success_outputs=success_outputs,
                       individuals_input_count=individuals_input_count,
                       repeat_count=repeat_count, tries_count=tries_count)

    final_tasks = node(task)

    # check final_tasks count
    # individuals that MockOperator returns
    _individuals_output_count = individuals_output_count or pop_size
    # individuals that MockOperator can get
    _individuals_input_count = individuals_input_count or pop_size
    # if there are repeats condition or task is divided due to
    # unappropriate individuals count (higher than individuals_input_count)
    # then incoming tasks are copied and divided
    incoming_tasks_count = ceil(pop_size / _individuals_input_count) * repeat_count
    # then only one task may be processed
    incoming_tasks_count -= 1
    processed_tasks_count = 1 * len(success_outputs)
    assert len(final_tasks) == (incoming_tasks_count + processed_tasks_count)


    # check tasks stage
    processed_task_stage = TaskStagesEnum.FINISH if success_outputs == [None] else TaskStagesEnum.SUCCESS
    if stage is TaskStagesEnum.INIT:
        assert sum(_task.stage is TaskStagesEnum.INIT for _task in final_tasks) == incoming_tasks_count
        assert sum(_task.stage is processed_task_stage for _task in final_tasks) == processed_tasks_count
    elif processed_task_stage is TaskStagesEnum.SUCCESS:
        assert all(_task.stage is TaskStagesEnum.SUCCESS for _task in final_tasks)
    else:
        assert sum(_task.stage is TaskStagesEnum.SUCCESS for _task in final_tasks) == incoming_tasks_count
        assert sum(_task.stage is processed_task_stage for _task in final_tasks) == processed_tasks_count

    # check left_tries
    assert all(_task.left_tries == tries_count for _task in final_tasks)

    # check prev and next nodes
    assert sum(_task.prev_stage_node == node_name for _task in final_tasks) == processed_tasks_count
    next_nodes = Counter(_task.next_stage_node for _task in final_tasks)
    if success_outputs == [None]:
        assert next_nodes[None] == len(final_tasks)
    else:
        assert set(next_nodes[name] for name in success_outputs) == {1}

    # check that processed task has correct individuals count
    assert all(len(_task.individuals) == (individuals_output_count or _individuals_input_count)
               for _task in final_tasks if _task.prev_stage_node == node.name)


@pytest.mark.parametrize(['success_outputs', 'left_tries',
                          'individuals_input_count', 'individuals_output_count',
                          'repeat_count', 'tries_count'],
                         product([[None], ['1', '2', '3']],  # success_outputs
                                 [1, 3],  # left_tries
                                 [1, 3, None],  # individuals_input_count
                                 [1, 3, None],  # individuals_output_count
                                 [1, 3],  # repeat_count
                                 [1, 3],  # tries_count
                                 ))
def test_genetic_node_with_nonfailed_task(success_outputs, left_tries, individuals_input_count,
                                          individuals_output_count, repeat_count, tries_count):
    pop_size = 10
    node_name = 'test'

    task = get_random_task(pop_size=pop_size, stage=TaskStagesEnum.FAIL, left_tries=left_tries)
    operator = MockOperator(success_prob=1, individuals_input_count=individuals_input_count,
                            individuals_output_count=individuals_output_count)
    node = GeneticNode(name=node_name, operator=operator, success_outputs=success_outputs,
                       individuals_input_count=individuals_input_count,
                       repeat_count=repeat_count, tries_count=tries_count)

    final_tasks = node(task)
