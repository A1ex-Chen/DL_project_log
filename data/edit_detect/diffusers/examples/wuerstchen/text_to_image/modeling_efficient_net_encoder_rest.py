import torch.nn as nn
from torchvision.models import efficientnet_v2_l, efficientnet_v2_s

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class EfficientNetEncoder(ModelMixin, ConfigMixin):
    @register_to_config
