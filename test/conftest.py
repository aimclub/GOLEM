import random
from unittest.mock import patch

import pytest

from golem.utilities.utils import set_random_seed


@pytest.fixture(autouse=True)
def stabilize_random():
    set_random_seed(42)

    def urandom_mock(n):
        return random.randbytes(n)

    # os.random is the source of random used in the uuid library
    # normally, it's „true“ random, but to stabilize tests,
    # it uses seeded `random` library
    with patch('os.urandom', urandom_mock):
        yield
