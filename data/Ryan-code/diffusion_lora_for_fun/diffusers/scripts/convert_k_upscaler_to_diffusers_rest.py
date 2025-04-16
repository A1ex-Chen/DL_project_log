import argparse

import huggingface_hub
import k_diffusion as K
import torch

from diffusers import UNet2DConditionModel


UPSCALER_REPO = "pcuenq/k-upscaler"
















if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()

    main(args)