#!/usr/bin/env python3
import torch

from diffusers import DiffusionPipeline


class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
