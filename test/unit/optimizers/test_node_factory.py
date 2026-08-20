from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory


def test_get_final_node_falls_back_to_get_node():
    factory = DefaultOptNodeFactory(available_node_types=['a', 'b'])
    for _ in range(20):
        node = factory.get_final_node()
        assert node is not None
        assert node.content['name'] in ('a', 'b')
