import itertools
from copy import deepcopy

import pytest

from test.unit.utils import graph_second, graph_first, graph_third, graph_fourth


@pytest.fixture()
def equality_cases():
    pairs = [[graph_first(), graph_first()], [graph_third(), graph_third()],
             [graph_fourth(), graph_fourth()]]

    # the following changes don't affect to graphs equality:
    for node_num in ((2, 1), (1, 2)):
        old_node = pairs[2][1].root_node.nodes_from[node_num[0]]
        new_node = deepcopy(pairs[2][0].root_node.nodes_from[node_num[1]])
        pairs[2][1].update_subtree(old_node, new_node)

    return pairs


@pytest.fixture()
def non_equality_cases():
    return list(itertools.combinations([graph_first(), graph_second(), graph_third()], 2))


@pytest.mark.parametrize('graph_fixture', ['equality_cases'])
def test_equality_cases(graph_fixture, request):
    list_graph_pairs = request.getfixturevalue(graph_fixture)
    for pair in list_graph_pairs:
        assert pair[0] == pair[1]
        assert pair[1] == pair[0]


@pytest.mark.parametrize('graph_fixture', ['non_equality_cases'])
def test_non_equality_cases(graph_fixture, request):
    list_graph_pairs = request.getfixturevalue(graph_fixture)
    for pair in list_graph_pairs:
        assert not pair[0] == pair[1]
        assert not pair[1] == pair[0]
