import os
from typing import Dict, Optional, Union

import pydantic
from azure.storage.blob import BlobServiceClient
from structlog import get_logger
from tenacity import retry

from modelkit.assets import errors
from modelkit.assets.drivers.abc import StorageDriver, StorageDriverSettings
from modelkit.assets.drivers.retry import retry_policy

logger = get_logger(__name__)

AZURE_RETRY_POLICY = retry_policy()


class AzureStorageDriverSettings(StorageDriverSettings):
    connection_string: Optional[str] = pydantic.Field(
        None,
        validation_alias=pydantic.AliasChoices(
            "connection_string", "AZURE_STORAGE_CONNECTION_STRING"
        ),
    )
    model_config = pydantic.ConfigDict(extra="ignore")


class AzureStorageDriver(StorageDriver):

    @staticmethod

    @retry(**AZURE_RETRY_POLICY)

    @retry(**AZURE_RETRY_POLICY)

    @retry(**AZURE_RETRY_POLICY)

    @retry(**AZURE_RETRY_POLICY)

    @retry(**AZURE_RETRY_POLICY)
