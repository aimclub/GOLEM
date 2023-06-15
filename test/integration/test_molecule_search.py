import pytest

from examples.molecule_search.experiment import molecule_search_setup, get_methane, get_all_mol_metrics
from examples.molecule_search.mol_adapter import MolAdapter


@pytest.mark.parametrize('metric_name, metric', get_all_mol_metrics().items())
def test_molecule_search_example(metric_name, metric):
    initial_molecule = get_methane()
    optimizer, objective = molecule_search_setup(num_iterations=20,
                                                 metrics=[metric_name],
                                                 initial_molecules=[initial_molecule])
    found_graphs = optimizer.optimise(objective)

    assert found_graphs is not None

    init_metric = metric(initial_molecule)
    found_graph = MolAdapter().restore(found_graphs[0])
    final_metric = metric(found_graph)

    assert final_metric < init_metric
