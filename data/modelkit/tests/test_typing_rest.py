import os

import mypy.api
import pytest

from tests import TEST_DIR




TEST_CASES = [
    ("predict_ok.py", False),
    ("predict_bad.py", True),
    ("predict_pydantic_ok.py", False),
    ("predict_pydantic_bad.py", True),
    ("predict_list.py", False),
    ("library_get_model_ko.py", True),
    ("library_get_model_ok.py", False),
    ("model_dependencies_ok.py", False),
    ("model_dependencies_ko_items.py", True),
]


@pytest.mark.parametrize("fn, raises", TEST_CASES)