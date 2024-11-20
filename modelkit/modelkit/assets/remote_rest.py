import datetime
import glob
import json
import os
import tempfile
import time
from typing import Any, Optional

import humanize
from dateutil import parser, tz
from structlog import get_logger

from modelkit.assets import errors
from modelkit.assets.drivers.abc import StorageDriver

try:
    from modelkit.assets.drivers.azure import (
        AzureStorageDriver,
        AzureStorageDriverSettings,
    )

    has_az = True
except ModuleNotFoundError:
    has_az = False
try:
    from modelkit.assets.drivers.gcs import GCSStorageDriver, GCSStorageDriverSettings

    has_gcs = True
except ModuleNotFoundError:
    has_gcs = False
from modelkit.assets.drivers.local import LocalStorageDriver, LocalStorageDriverSettings

try:
    from modelkit.assets.drivers.s3 import S3StorageDriver, S3StorageDriverSettings

    has_s3 = True
except ModuleNotFoundError:
    has_s3 = False
from modelkit.assets.settings import AssetSpec
from modelkit.utils.logging import ContextualizedLogging

logger = get_logger(__name__)




class UnknownDriverError(Exception):
    pass


class DriverNotInstalledError(Exception):
    pass


class NoConfiguredProviderError(Exception):
    pass


class StorageProvider:
    driver: StorageDriver
    force_download: bool
    prefix: str
    timeout: int










