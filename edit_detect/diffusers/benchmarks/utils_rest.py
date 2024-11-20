import argparse
import csv
import gc
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
import torch.utils.benchmark as benchmark


GITHUB_SHA = os.getenv("GITHUB_SHA", None)
BENCHMARK_FIELDS = [
    "pipeline_cls",
    "ckpt_id",
    "batch_size",
    "num_inference_steps",
    "model_cpu_offload",
    "run_compile",
    "time (secs)",
    "memory (gbs)",
    "actual_gpu_memory (gbs)",
    "github_sha",
]

PROMPT = "ghibli style, a fantasy landscape with castles"
BASE_PATH = os.getenv("BASE_PATH", ".")
TOTAL_GPU_MEMORY = float(os.getenv("TOTAL_GPU_MEMORY", torch.cuda.get_device_properties(0).total_memory / (1024**3)))

REPO_ID = "diffusers/benchmarks"
FINAL_CSV_FILE = "collated_results.csv"


@dataclass
class BenchmarkInfo:
    time: float
    memory: float











