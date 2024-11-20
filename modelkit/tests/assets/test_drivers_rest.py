import os
import pickle
import tempfile
from typing import Optional

from modelkit.assets.drivers.abc import StorageDriver, StorageDriverSettings
from modelkit.assets.drivers.local import LocalStorageDriver
from tests import TEST_DIR
from tests.conftest import skip_unless










@skip_unless("ENABLE_GCS_TEST", "True")


@skip_unless("ENABLE_GCS_TEST", "True")


@skip_unless("ENABLE_S3_TEST", "True")


@skip_unless("ENABLE_S3_TEST", "True")


@skip_unless("ENABLE_AZ_TEST", "True")


@skip_unless("ENABLE_AZ_TEST", "True")











    # the storage provider should not build the client
    # since passed at instantiation
    settings = StorageDriverSettings(bucket="bucket")
    driver = MockedDriver(settings, client={"built": False, "passed": True})
    assert settings.lazy_driver is False
    assert driver._client is not None
    assert driver._client == driver.client == {"built": False, "passed": True}

    # the storage provider should build the client at init
    driver = MockedDriver(settings)
    assert settings.lazy_driver is False
    assert driver._client is not None
    assert driver._client == driver.client == {"built": True, "passed": False}

    monkeypatch.setenv("MODELKIT_LAZY_DRIVER", True)
    # the storage provider should not build the client eagerly nor store it
    # since MODELKIT_LAZY_DRIVER is set
    settings = StorageDriverSettings(bucket="bucket")
    driver = MockedDriver(settings)
    assert settings.lazy_driver is True
    assert driver._client is None
    # the storage provider builds it on-the-fly when accessed via the `client` property
    assert driver.client == {"built": True, "passed": False}
    # but does not store it
    assert driver._client is None

    # the storage provider should not build any client but use the one passed
    # at instantiation
    driver = MockedDriver(settings, client={"built": False, "passed": True})
    assert driver._client is not None
    assert driver._client == driver.client == {"built": False, "passed": True}