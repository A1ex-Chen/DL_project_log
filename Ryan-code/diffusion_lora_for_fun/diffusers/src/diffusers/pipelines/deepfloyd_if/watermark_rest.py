from typing import List

import PIL.Image
import torch
from PIL import Image

from ...configuration_utils import ConfigMixin
from ...models.modeling_utils import ModelMixin
from ...utils import PIL_INTERPOLATION


class IFWatermarker(ModelMixin, ConfigMixin):
