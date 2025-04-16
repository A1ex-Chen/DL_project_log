import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, ModelFilter

import diffusers


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
ALWAYS_TEST_PIPELINE_MODULES = [
    "controlnet",
    "stable_diffusion",
    "stable_diffusion_2",
    "stable_diffusion_xl",
    "stable_diffusion_adapter",
    "deepfloyd_if",
    "ip_adapters",
    "kandinsky",
    "kandinsky2_2",
    "text_to_video_synthesis",
    "wuerstchen",
]
PIPELINE_USAGE_CUTOFF = int(os.getenv("PIPELINE_USAGE_CUTOFF", 50000))

logger = logging.getLogger(__name__)
api = HfApi()
filter = ModelFilter(library="diffusers")










if __name__ == "__main__":
    main()