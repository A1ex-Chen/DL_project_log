import abc
import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional

import cachetools
import cachetools.keys
import pydantic

import modelkit
from modelkit.core.types import ItemType
from modelkit.utils.redis import connect_redis


@dataclass
class CacheItem(Generic[ItemType]):
    item: Optional[ItemType] = None
    cache_key: Optional[bytes] = None
    cache_value: Optional[Any] = None
    missing: bool = True


class Cache(abc.ABC):
    @abc.abstractmethod

    @abc.abstractmethod

    @abc.abstractmethod


class RedisCache(Cache):





class NativeCache(Cache):
    NATIVE_CACHE_IMPLEMENTATIONS = {
        "LFU": cachetools.LFUCache,
        "LRU": cachetools.LRUCache,
        "RR": cachetools.RRCache,
    }



