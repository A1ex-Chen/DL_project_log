import os

import pytest

from modelkit.assets import errors
from tests.conftest import skip_unless






@skip_unless("ENABLE_GCS_TEST", "True")


@skip_unless("ENABLE_S3_TEST", "True")


@skip_unless("ENABLE_AZ_TEST", "True")