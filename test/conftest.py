import random
from unittest.mock import patch

import pytest

from golem.utilities.utilities import urandom_mock
from golem.utilities.utils import set_random_seed


@pytest.fixture(autouse=True)
def stabilize_random():
    set_random_seed(42)

    with patch('os.urandom', urandom_mock):
        yield
