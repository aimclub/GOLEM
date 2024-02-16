import os.path
from pathlib import Path

import pytest
from mabwiser.mab import MAB

from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent


@pytest.mark.parametrize('path_to_save, file_name',
                         [(os.path.join(Path(__file__).parent, 'test_mab.pkl'), 'test_mab.pkl'),
                          (os.path.join(Path(__file__).parent), '0_mab.pkl')])
def test_save_mab(path_to_save, file_name):
    """ Tests if MAB is saved with specifying file_nam and without. """
    mab = MultiArmedBanditAgent(actions=['0', '1', '2'],
                                n_jobs=1,
                                path_to_save=path_to_save)
    mab.save()
    assert file_name in os.listdir(Path(__file__).parent)
    os.remove(path_to_save if path_to_save.endswith('pkl') else os.path.join(path_to_save, file_name))


def test_load_mab():
    """ Tests if MAB is loaded. """
    file_name = 'test_mab.pkl'
    path_to_load = os.path.join(Path(__file__).parent, file_name)
    # save mab to load it later
    mab = MultiArmedBanditAgent(actions=['0', '1', '2'],
                                n_jobs=1,
                                path_to_save=path_to_load)
    mab.save()

    loaded_mab = MultiArmedBanditAgent.load(path=path_to_load)
    assert isinstance(loaded_mab, MultiArmedBanditAgent)

    assert isinstance(loaded_mab._agent, MAB)
    assert loaded_mab.__eq__(mab)
    assert loaded_mab.actions == mab.actions
    assert loaded_mab._enable_logging == mab._enable_logging
    assert loaded_mab._path_to_save == mab._path_to_save

    os.remove(path_to_load)
