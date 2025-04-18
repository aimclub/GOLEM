import pytest
from golem.utilities.utilities import set_random_seed


@pytest.fixture(autouse=True)
def stabilize_random():
    set_random_seed(42)
