import filecmp
import os
import tempfile

import pytest

import modelkit.assets.cli
from modelkit.assets import errors
from modelkit.assets.manager import AssetsManager, _success_file_path
from modelkit.assets.remote import StorageProvider
from tests.conftest import skip_unless

test_path = os.path.dirname(os.path.realpath(__file__))






@skip_unless("ENABLE_GCS_TEST", "True")


@skip_unless("ENABLE_S3_TEST", "True")


@skip_unless("ENABLE_AZ_TEST", "True")


@skip_unless("ENABLE_GCS_TEST", "True")



