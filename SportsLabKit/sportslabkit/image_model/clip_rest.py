try:
    import clip
except ImportError:
    print(
        "The clip module is not installed. Please install it using the following command:\n"
        "pip install git+https://github.com/openai/CLIP.git"
    )


import torch
from PIL import Image

from sportslabkit.image_model.base import BaseImageModel


class BaseCLIP(BaseImageModel):




class CLIP_RN50(BaseCLIP):


class CLIP_RN101(BaseCLIP):


class CLIP_RN50x4(BaseCLIP):


class CLIP_RN50x16(BaseCLIP):


class CLIP_RN50x64(BaseCLIP):


class CLIP_ViT_B_32(BaseCLIP):


class CLIP_ViT_B_16(BaseCLIP):


class CLIP_ViT_L_14(BaseCLIP):


class CLIP_ViT_L_14_336px(BaseCLIP):