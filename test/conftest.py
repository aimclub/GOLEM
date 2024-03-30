import random
from unittest.mock import patch

import pytest

from golem.utilities.utils import set_random_seed


@pytest.fixture(autouse=True)
def stabilize_random():
    set_random_seed(42)

    def urandom_mock(n):
        return random.randbytes(n)

    with patch('os.urandom', urandom_mock):
        yield
