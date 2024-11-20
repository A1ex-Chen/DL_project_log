# coding=utf-8
# Copyright 2020 Hugging Face
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

import time
from typing import Optional

import IPython.display as disp

from ..trainer_callback import TrainerCallback
from ..trainer_utils import EvaluationStrategy








class NotebookProgressBar:
    """
    A progress par for display in a notebook.

    Class attributes (overridden by derived classes)

        - **warmup** (:obj:`int`) -- The number of iterations to do at the beginning while ignoring
          :obj:`update_every`.
        - **update_every** (:obj:`float`) -- Since calling the time takes some time, we only do it every presumed
          :obj:`update_every` seconds. The progress bar uses the average time passed up until now to guess the next
          value for which it will call the update.

    Args:
        total (:obj:`int`):
            The total number of iterations to reach.
        prefix (:obj:`str`, `optional`):
            A prefix to add before the progress bar.
        leave (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to leave the progress bar once it's completed. You can always call the
            :meth:`~transformers.utils.notebook.NotebookProgressBar.close` method to make the bar disappear.
        parent (:class:`~transformers.notebook.NotebookTrainingTracker`, `optional`):
            A parent object (like :class:`~transformers.utils.notebook.NotebookTrainingTracker`) that spawns progress
            bars and handle their display. If set, the object passed must have a :obj:`display()` method.
        width (:obj:`int`, `optional`, defaults to 300):
            The width (in pixels) that the bar will take.

    Example::

        import time

        pbar = NotebookProgressBar(100)
        for val in range(100):
            pbar.update(val)
            time.sleep(0.07)
        pbar.update(100)
    """

    warmup = 5
    update_every = 0.2







class NotebookTrainingTracker(NotebookProgressBar):
    """
    An object tracking the updates of an ongoing training with progress bars and a nice table reporting metrics.

    Args:

        num_steps (:obj:`int`): The number of steps during training.
        column_names (:obj:`List[str]`, `optional`):
            The list of column names for the metrics table (will be inferred from the first call to
            :meth:`~transformers.utils.notebook.NotebookTrainingTracker.write_line` if not set).
    """







class NotebookProgressCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that displays the progress of training or evaluation, optimized for
    Jupyter Notebooks or Google colab.
    """






