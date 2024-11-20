import collections
import inspect
import os
import re
import torch
from deepview_profile.utils import model_location_patterns

SourceLocation = collections.namedtuple(
    "SourceLocation", ["file_path", "line_number", "module_id"]
)


class CallStack:

    @staticmethod