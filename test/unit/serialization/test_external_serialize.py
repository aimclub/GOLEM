import os.path
from pathlib import Path

import pytest

from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.paths import project_root
from golem.visualisation.opt_viz import PlotTypesEnum, OptHistoryVisualizer


@pytest.mark.parametrize('history_path', [
    'test/data/history_composite_bn_healthcare.json',
])
def test_external_history_load(history_path):
    """The idea is that external histories must be loadable by GOLEM.
    External histories are those by projects that depend on GOLEM, e.g. BAMT.

    This is needed so that GOLEM could be used as a stand-alone
    analytic tool for external histories. Or, or example, external histories
    could be used in FEDOT.Web that depends only on GOLEM.
    """
    history_path = project_root() / history_path

    assert os.path.exists(history_path)

    # Expect load without errors
    history: OptHistory = OptHistory.load(history_path)

    for plot_type in PlotTypesEnum:
        save_path = Path(plot_type.name)
        save_path = save_path.with_suffix('.gif') if plot_type is PlotTypesEnum.operations_animated_bar \
            else save_path.with_suffix('.png')
        visualizer = OptHistoryVisualizer(history)
        visualization = plot_type.value(visualizer.history, visualizer.visuals_params)
        visualization.visualize(save_path=str(save_path), best_fraction=0.1, dpi=100)

    assert history is not None
    assert len(history.individuals) > 0
