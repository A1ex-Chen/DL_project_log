# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

MB2B = 2 ** 20
B2MB = 1 / MB2B
FLUSH_THRESHOLD_B = 256 * MB2B




class NpzWriter:
    """
    Dumps dicts of numpy arrays into npz files

    It can/shall be used as context manager:
    ```
    with OutputWriter('mydir') as writer:
        writer.write(outputs={'classes': np.zeros(8), 'probs': np.zeros((8, 4))},
                     labels={'classes': np.zeros(8)},
                     inputs={'input': np.zeros((8, 240, 240, 3)})
    ```

    ## Variable size data

    Only dynamic of last axis is handled. Data is padded with np.nan value.
    Also each generated file may have different size of dynamic axis.
    """


    @property





