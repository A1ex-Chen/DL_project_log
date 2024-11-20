import os
import shutil
import stat

import pytest

from modelkit.assets import errors
from modelkit.assets.manager import AssetsManager, _fetch_local_version
from modelkit.assets.remote import StorageProvider
from modelkit.assets.settings import AssetSpec
from tests import TEST_DIR
from tests.assets.test_versioning import test_versioning




@pytest.mark.parametrize(
    "v00, v01, v11, v10, versioning",
    [
        ("0.0", "0.1", "1.0", "1.1", None),
        ("0.0", "0.1", "1.0", "1.1", "major_minor"),
        (
            "0000-00-00T00-00-00Z",
            "0000-00-00T01-00-00Z",
            "0000-00-00T10-00-00Z",
            "0000-00-00T11-00-00Z",
            "simple_date",
        ),
    ],
)


@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)




@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)


@pytest.mark.parametrize(*test_versioning.INIT_VERSIONING_PARAMETRIZE)


@pytest.mark.parametrize(*test_versioning.INIT_VERSIONING_PARAMETRIZE)


@pytest.mark.parametrize(*test_versioning.INIT_VERSIONING_PARAMETRIZE)

