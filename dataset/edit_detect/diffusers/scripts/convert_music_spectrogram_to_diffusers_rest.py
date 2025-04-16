#!/usr/bin/env python3
import argparse
import os

import jax as jnp
import numpy as onp
import torch
import torch.nn as nn
from music_spectrogram_diffusion import inference
from t5x import checkpoints

from diffusers import DDPMScheduler, OnnxRuntimeModel, SpectrogramDiffusionPipeline
from diffusers.pipelines.spectrogram_diffusion import SpectrogramContEncoder, SpectrogramNotesEncoder, T5FilmDecoder


MODEL = "base_with_context"










if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", default=None, type=str, required=True, help="Path to the converted model.")
    parser.add_argument(
        "--save", default=True, type=bool, required=False, help="Whether to save the converted model or not."
    )
    parser.add_argument(
        "--checkpoint_path",
        default=f"{MODEL}/checkpoint_500000",
        type=str,
        required=False,
        help="Path to the original jax model checkpoint.",
    )
    args = parser.parse_args()

    main(args)