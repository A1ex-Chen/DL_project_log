# Copyright (c) Open-MMLab. All rights reserved.
from pathlib import Path
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler()
}











    return wrap
