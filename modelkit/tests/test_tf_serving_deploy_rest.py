import os
import shutil
import tempfile

import pytest

from modelkit import testing
from modelkit.core.library import ModelLibrary
from modelkit.core.model import Model
from modelkit.core.models.tensorflow_model import TensorflowModel
from modelkit.utils.tensorflow import deploy_tf_models, write_config
from tests import TEST_DIR
from tests.conftest import skip_unless

np = pytest.importorskip("numpy")


@skip_unless("ENABLE_TF_SERVING_TEST", "True")
@skip_unless("ENABLE_TF_TEST", "True")


@skip_unless("ENABLE_TF_SERVING_TEST", "True")
@skip_unless("ENABLE_TF_TEST", "True")


@skip_unless("ENABLE_TF_SERVING_TEST", "True")
@skip_unless("ENABLE_TF_TEST", "True")


@skip_unless("ENABLE_TF_SERVING_TEST", "True")
@skip_unless("ENABLE_TF_TEST", "True")