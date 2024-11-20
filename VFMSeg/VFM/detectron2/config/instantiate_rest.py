# Copyright (c) Facebook, Inc. and its affiliates.
import dataclasses
import logging
from collections import abc
from typing import Any

from detectron2.utils.registry import _convert_target_to_string, locate

__all__ = ["dump_dataclass", "instantiate"]



