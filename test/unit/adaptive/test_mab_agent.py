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
    mab = MultiArmedBanditAgent(actions=[0, 1, 2],
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
    mab = MultiArmedBanditAgent(actions=[0, 1, 2],
                                n_jobs=1,
                                path_to_save=path_to_load)
    mab.save()

    loaded_mab = MultiArmedBanditAgent.load(path=path_to_load)
    assert isinstance(loaded_mab, MultiArmedBanditAgent)

    assert isinstance(loaded_mab._agent, MAB)
    # NB: neither MultiArmedBanditAgent nor MAB define __eq__, so a direct
    # __eq__ call returns NotImplemented - truthy before Python 3.14, a
    # TypeError in a boolean context since; the field asserts below are the
    # actual equality check
    assert loaded_mab.actions == mab.actions
    assert loaded_mab._enable_logging == mab._enable_logging
    assert loaded_mab._path_to_save == mab._path_to_save

    os.remove(path_to_load)


def test_save_mab_file_numbering(tmp_path):
    """ Tests that the next saved file gets max existing number + 1,
    including numbers with more than one digit, and that files
    with non-numeric prefixes are ignored rather than breaking the save. """
    for name in ['0_mab.pkl', '2_mab.pkl', '10_mab.pkl', '_mab.pkl', 'other.txt']:
        (tmp_path / name).touch()

    mab = MultiArmedBanditAgent(actions=[0, 1, 2], n_jobs=1)
    mab.save(path_to_save=str(tmp_path))

    assert '11_mab.pkl' in os.listdir(tmp_path)
