import abc
from typing import Any, Dict, Iterator, Optional, Union

import pydantic

from modelkit.core.settings import ModelkitSettings


class StorageDriverSettings(ModelkitSettings):
    bucket: str = pydantic.Field(
        ..., validation_alias=pydantic.AliasChoices("bucket", "MODELKIT_STORAGE_BUCKET")
    )
    lazy_driver: bool = pydantic.Field(
        False,
        validation_alias=pydantic.AliasChoices("lazy_driver", "MODELKIT_LAZY_DRIVER"),
    )
    model_config = pydantic.ConfigDict(extra="allow")


class StorageDriver(abc.ABC):

    @abc.abstractmethod

    @abc.abstractmethod

    @abc.abstractmethod

    @abc.abstractmethod

    @abc.abstractmethod

    @abc.abstractmethod

    @property

    @staticmethod
    @abc.abstractmethod