"""
Utilities for working with package versions
"""

import operator
import re
import sys
from typing import Optional

from packaging import version

import pkg_resources


ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}





