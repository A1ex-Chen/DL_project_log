import pydantic
import pytest

from modelkit.assets.drivers.abc import StorageDriverSettings
from modelkit.assets.drivers.azure import AzureStorageDriverSettings
from modelkit.assets.drivers.gcs import GCSStorageDriverSettings
from modelkit.assets.drivers.local import LocalStorageDriverSettings
from modelkit.assets.drivers.s3 import S3StorageDriverSettings
from modelkit.core.settings import (
    LibrarySettings,
    ModelkitSettings,
    NativeCacheSettings,
    RedisSettings,
)




@pytest.mark.parametrize(
    "Settings",
    [
        StorageDriverSettings,
        GCSStorageDriverSettings,
        AzureStorageDriverSettings,
        S3StorageDriverSettings,
        LocalStorageDriverSettings,
    ],
)

