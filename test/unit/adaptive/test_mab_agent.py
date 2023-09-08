import os.path
from pathlib import Path

from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent


def test_save_mab():
    """ Tests if MAB is saved. """
    file_name = 'test_mab.pkl'
    path_to_save = os.path.join(Path(__file__).parent, file_name)
    mab = MultiArmedBanditAgent(actions=[0, 1, 2],
                                n_jobs=1,
                                path_to_save=path_to_save)
    mab.save()
    assert file_name in os.listdir(Path(__file__).parent)


def test_load_mab():
    """ Tests if MAB is loaded. """
    file_name = 'test_mab.pkl'
    path_to_load = os.path.join(Path(__file__).parent, file_name)
    mab = MultiArmedBanditAgent.load(path=path_to_load)
    assert isinstance(mab, MultiArmedBanditAgent)
    os.remove(path_to_load)
