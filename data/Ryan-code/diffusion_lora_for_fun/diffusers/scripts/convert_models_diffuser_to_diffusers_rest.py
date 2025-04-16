import json
import os

import torch

from diffusers import UNet1DModel


os.makedirs("hub/hopper-medium-v2/unet/hor32", exist_ok=True)
os.makedirs("hub/hopper-medium-v2/unet/hor128", exist_ok=True)

os.makedirs("hub/hopper-medium-v2/value_function", exist_ok=True)






if __name__ == "__main__":
    unet(32)
    # unet(128)
    value_function()