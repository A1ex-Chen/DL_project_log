import os
from typing import Dict, Optional, Union

import boto3
import botocore
import pydantic
from structlog import get_logger
from tenacity import retry

from modelkit.assets import errors
from modelkit.assets.drivers.abc import StorageDriver, StorageDriverSettings
from modelkit.assets.drivers.retry import retry_policy

logger = get_logger(__name__)

S3_RETRY_POLICY = retry_policy(botocore.exceptions.ClientError)


class S3StorageDriverSettings(StorageDriverSettings):
    aws_access_key_id: Optional[str] = pydantic.Field(
        None,
        validation_alias=pydantic.AliasChoices(
            "aws_access_key_id", "AWS_ACCESS_KEY_ID"
        ),
    )
    aws_secret_access_key: Optional[str] = pydantic.Field(
        None,
        validation_alias=pydantic.AliasChoices(
            "aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"
        ),
    )
    aws_default_region: Optional[str] = pydantic.Field(
        None,
        validation_alias=pydantic.AliasChoices(
            "aws_default_region", "AWS_DEFAULT_REGION"
        ),
    )
    aws_session_token: Optional[str] = pydantic.Field(
        None,
        validation_alias=pydantic.AliasChoices(
            "aws_session_token", "AWS_SESSION_TOKEN"
        ),
    )
    s3_endpoint: Optional[str] = pydantic.Field(
        None, validation_alias=pydantic.AliasChoices("s3_endpoint", "S3_ENDPOINT")
    )
    aws_kms_key_id: Optional[str] = pydantic.Field(
        None, validation_alias=pydantic.AliasChoices("aws_kms_key_id", "AWS_KMS_KEY_ID")
    )
    model_config = pydantic.ConfigDict(extra="ignore")


class S3StorageDriver(StorageDriver):

    @staticmethod

    @retry(**S3_RETRY_POLICY)

    @retry(**S3_RETRY_POLICY)

    @retry(**S3_RETRY_POLICY)

    @retry(**S3_RETRY_POLICY)

    @retry(**S3_RETRY_POLICY)

