# Copyright (c) Open-MMLab. All rights reserved.
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .base import BaseFileHandler  # isort:skip


class YamlHandler(BaseFileHandler):


