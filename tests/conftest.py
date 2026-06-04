"""Pytest fixtures."""

import pytest
from _phantom import make_phantom


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()
