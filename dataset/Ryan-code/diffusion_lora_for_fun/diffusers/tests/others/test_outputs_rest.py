import pickle as pkl
import unittest
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils.outputs import BaseOutput
from diffusers.utils.testing_utils import require_torch


@dataclass
class CustomOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]


class ConfigTester(unittest.TestCase):



    @require_torch