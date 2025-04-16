#  Copyright 2024 Custom Diffusion authors. All rights reserved.
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
import argparse
import os
from io import BytesIO
from pathlib import Path

import requests
from clip_retrieval.clip_client import ClipClient
from PIL import Image
from tqdm import tqdm






if __name__ == "__main__":
    args = parse_args()
    retrieve(args.class_prompt, args.class_data_dir, args.num_class_images)