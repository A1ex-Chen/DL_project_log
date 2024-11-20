import os
from typing import Dict, Optional, Union

import pydantic
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import storage
from google.cloud.storage import Client
from structlog import get_logger
from tenacity import retry

from modelkit.assets import errors
from modelkit.assets.drivers.abc import StorageDriver, StorageDriverSettings
from modelkit.assets.drivers.retry import retry_policy

logger = get_logger(__name__)

GCS_RETRY_POLICY = retry_policy(GoogleAPIError)


class GCSStorageDriverSettings(StorageDriverSettings):
    service_account_path: Optional[str] = pydantic.Field(
        None,
        validation_alias=pydantic.AliasChoices(
            "service_account_path", "GOOGLE_APPLICATION_CREDENTIALS"
        ),
    )
    model_config = pydantic.ConfigDict(extra="ignore")


class GCSStorageDriver(StorageDriver):

    @staticmethod

    @retry(**GCS_RETRY_POLICY)

    @retry(**GCS_RETRY_POLICY)

    @retry(**GCS_RETRY_POLICY)

    @retry(**GCS_RETRY_POLICY)

    @retry(**GCS_RETRY_POLICY)
