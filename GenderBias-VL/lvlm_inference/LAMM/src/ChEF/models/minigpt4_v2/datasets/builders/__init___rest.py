"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from .base_dataset_builder import load_dataset_config
from .image_text_pair_builder import (
    CCSBUBuilder,
    CCSBUAlignBuilder
)
from ...common.registry import registry

__all__ = [
    "CCSBUBuilder",
    "CCSBUAlignBuilder"
]




class DatasetZoo:



dataset_zoo = DatasetZoo()