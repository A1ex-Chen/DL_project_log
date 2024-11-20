import pytest
import pickle
from utils import DeepviewSession, BackendContext
from google.protobuf.json_format import MessageToDict
from config_params import TestConfig
import os

REPS = 2
NUM_EXPECTED_MESSAGES = 6




config = TestConfig()

tests = list()
for model_name in config["model_names_from_examples"]:
    dir_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        model_name,
    )
    tests.append((model_name, dir_path))


@pytest.mark.parametrize("test_name, entry_point", tests)