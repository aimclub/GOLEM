from pathlib import Path

import pytest

from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.paths import project_root
from golem.visualisation.opt_viz import PlotTypesEnum, OptHistoryVisualizer


@pytest.mark.parametrize('history_path', [
    'test/data/history_composite_bn_healthcare.json',
])
@pytest.mark.parametrize('plot_type', PlotTypesEnum)
def test_visualizations_for_external_history(tmp_path, history_path, plot_type):
    history_path = project_root() / history_path
    history = OptHistory.load(history_path)
    save_path = Path(tmp_path, plot_type.name)
    save_path = save_path.with_suffix('.gif') if plot_type is PlotTypesEnum.operations_animated_bar \
        else save_path.with_suffix('.png')
    visualizer = OptHistoryVisualizer(history)
    visualization = plot_type.value(visualizer.history, visualizer.visuals_params)
    visualization.visualize(save_path=str(save_path), best_fraction=0.1, dpi=100)
    if plot_type is not PlotTypesEnum.fitness_line_interactive:
        assert save_path.exists()