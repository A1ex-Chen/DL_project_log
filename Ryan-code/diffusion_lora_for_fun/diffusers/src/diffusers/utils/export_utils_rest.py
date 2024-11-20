import io
import random
import struct
import tempfile
from contextlib import contextmanager
from typing import List, Union

import numpy as np
import PIL.Image
import PIL.ImageOps

from .import_utils import (
    BACKENDS_MAPPING,
    is_opencv_available,
)
from .logging import get_logger


global_rng = random.Random()

logger = get_logger(__name__)


@contextmanager







