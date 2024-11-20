def pytest_addoption(parser):
    from diffusers.utils.testing_utils import pytest_addoption_shared
    pytest_addoption_shared(parser)
