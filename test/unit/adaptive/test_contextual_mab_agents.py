import os
import random
from importlib.util import find_spec

import pytest

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BanditNetworkxAdapter
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.experience_buffer import ExperienceBuffer
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import Individual

adapter = BanditNetworkxAdapter()
available_operations = ['1', '2', '3', '4', '5']


def get_small_and_large_graphs():
    """ Generates and adapts two graphs:
    a 'small' graph with random size from 1 to 10
    a 'large' graph with random size from 100 to 200. """

    small_graph_size = random.randint(1, 10)
    large_graph_size = random.randint(100, 200)

    small_graph = generate_labeled_graph('tree', small_graph_size, node_labels=available_operations)
    small_graph = adapter.adapt(small_graph)
    large_graph = generate_labeled_graph('tree', large_graph_size, node_labels=available_operations)
    large_graph = adapter.adapt(large_graph)

    return small_graph, large_graph


@pytest.mark.parametrize('context_agent, context_size',
                         [(ContextAgentTypeEnum.operations_quantity, len(available_operations)),
                          (ContextAgentTypeEnum.adjacency_matrix, len(available_operations) ** 2),
                          pytest.param(ContextAgentTypeEnum.feather_graph, 500,
                                       marks=pytest.mark.skipif(find_spec('karateclub') is None,
                                                                reason='karateclub is not available '
                                                                       '(unsupported on Python 3.11+)')),
                          (ContextAgentTypeEnum.labeled_edges, 100),
                          (ContextAgentTypeEnum.nodes_num, 1)])
def test_contextual_mab_agents(context_agent, context_size):
    """ Checks the correctness of context size, returned by context agents of Contextual MAB Agents. """

    cmab_agent = ContextualMultiArmedBanditAgent(actions=MutationTypesEnum,
                                                 available_operations=available_operations,
                                                 context_agent_type=context_agent)

    neural_cmab_agent = NeuralContextualMultiArmedBanditAgent(actions=MutationTypesEnum,
                                                              available_operations=available_operations,
                                                              context_agent_type=context_agent)

    small_graph, large_graph = get_small_and_large_graphs()

    assert cmab_agent.get_context(small_graph).shape[1] == context_size
    assert cmab_agent.get_context(large_graph).shape[1] == context_size
    assert neural_cmab_agent.get_context(small_graph).shape[1] == context_size
    assert neural_cmab_agent.get_context(large_graph).shape[1] == context_size


def test_contextual_mab_saves_state_on_partial_fit(tmp_path):
    """ Contextual MAB agent with a specified path_to_save must save its state
    after each partial_fit, so that it can be restored later. """
    cmab_agent = ContextualMultiArmedBanditAgent(actions=[0, 1, 2],
                                                 available_operations=available_operations,
                                                 context_agent_type=ContextAgentTypeEnum.nodes_num,
                                                 path_to_save=str(tmp_path))

    experience = ExperienceBuffer()
    experience.collect_experience(Individual(OptGraph(OptNode('1'))), action=1, reward=0.5)
    cmab_agent.partial_fit(experience)

    assert any(name.endswith('_mab.pkl') for name in os.listdir(tmp_path))
