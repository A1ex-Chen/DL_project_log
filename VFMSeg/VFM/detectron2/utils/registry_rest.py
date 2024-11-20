# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any
import pydoc
from fvcore.common.registry import Registry  # for backward compatibility.

"""
``Registry`` and `locate` provide ways to map a string (typically found
in config files) to callable objects.
"""

__all__ = ["Registry", "locate"]



