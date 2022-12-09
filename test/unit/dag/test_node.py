from golem.core.dag.linked_graph_node import LinkedGraphNode


def test_node_description():
    # given
    operation_type = 'logit'
    test_model_node = LinkedGraphNode(dict(name=operation_type))
    expected_node_description = f'n_{operation_type}'

    # when
    actual_node_description = test_model_node.description()

    # then
    assert actual_node_description == expected_node_description


def test_node_description_with_params():
    # given
    operation_type = 'logit'
    params = {'some_param': 10}
    test_model_node = LinkedGraphNode(dict(name=operation_type, params=params))
    expected_node_description = f'n_{operation_type}_{params}'

    # when
    actual_node_description = test_model_node.description()

    # then
    assert actual_node_description == expected_node_description
