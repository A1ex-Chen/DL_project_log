# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import numpy as np  # noqa: E402

from ....configuration_utils import ConfigMixin, register_to_config
from ....schedulers.scheduling_utils import SchedulerMixin


try:
    import librosa  # noqa: E402

    _librosa_can_be_imported = True
    _import_error = ""
except Exception as e:
    _librosa_can_be_imported = False
    _import_error = (
        f"Cannot import librosa because {e}. Make sure to correctly install librosa to be able to install it."
    )


from PIL import Image  # noqa: E402


class Mel(ConfigMixin, SchedulerMixin):
    """
    Parameters:
        x_res (`int`):
            x resolution of spectrogram (time).
        y_res (`int`):
            y resolution of spectrogram (frequency bins).
        sample_rate (`int`):
            Sample rate of audio.
        n_fft (`int`):
            Number of Fast Fourier Transforms.
        hop_length (`int`):
            Hop length (a higher number is recommended if `y_res` < 256).
        top_db (`int`):
            Loudest decibel value.
        n_iter (`int`):
            Number of iterations for Griffin-Lim Mel inversion.
    """

    config_name = "mel_config.json"

    @register_to_config






