import glob
import os
import shutil
from typing import Dict, Optional, Union

import pydantic
from structlog import get_logger

from modelkit.assets import errors
from modelkit.assets.drivers.abc import StorageDriver, StorageDriverSettings

logger = get_logger(__name__)


class LocalStorageDriverSettings(StorageDriverSettings):
    model_config = pydantic.ConfigDict(extra="ignore")


class LocalStorageDriver(StorageDriver):

    @staticmethod






