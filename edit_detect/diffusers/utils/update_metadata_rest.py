# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility that updates the metadata of the Diffusers library in the repository `huggingface/diffusers-metadata`.

Usage for an update (as used by the GitHub action `update_metadata`):

```bash
python utils/update_metadata.py
```

Script modified from:
https://github.com/huggingface/transformers/blob/main/utils/update_metadata.py
"""
import argparse
import os
import tempfile

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, upload_folder

from diffusers.pipelines.auto_pipeline import (
    AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
    AUTO_INPAINT_PIPELINES_MAPPING,
    AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
)


PIPELINE_TAG_JSON = "pipeline_tags.json"






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit_sha", default=None, type=str, help="The sha of the commit going with this update.")
    args = parser.parse_args()

    update_metadata(args.commit_sha)