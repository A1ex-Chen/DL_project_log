import dataclasses
import inspect
import os

import pydantic
import pydantic.generics

from modelkit.core.library import ModelLibrary
from modelkit.core.model import Model
from modelkit.core.model_configuration import configure
from modelkit.testing.reference import ReferenceJson

try:
    import numpy as np

    has_numpy = True
except ModuleNotFoundError:  # pragma: no cover
    has_numpy = False


@dataclasses.dataclass
class JSONTestResult:
    fn: str





    # in order for the above functions to be collected by pytest, add them
    # to the caller's local variables under their desired names
    frame = inspect.currentframe().f_back
    frame.f_locals[test_name] = test_function


def modellibrary_fixture(
    # arguments passed directly to ModelLibrary
    settings=None,
    assetsmanager_settings=None,
    configuration=None,
    models=None,
    required_models=None,
    #  fixture name
    fixture_name="testing_model_library",
    necessary_fixtures=None,
    fixture_scope="session",
):
    import pytest

    #  create a named fixture with the ModelLibrary
    @pytest.fixture(name=fixture_name, scope=fixture_scope)

    # in order for the above functions to be collected by pytest, add them
    # to the caller's local variables under their desired names
    frame = inspect.currentframe().f_back
    frame.f_locals[fixture_name] = fixture_function