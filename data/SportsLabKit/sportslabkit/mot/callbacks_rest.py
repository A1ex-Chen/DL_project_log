"""Defines the Callback base class and utility decorators for use with the Trainer class.

The Callback class provides a dynamic way to hook into various stages of the Trainer's operations.
It uses Python's __getattr__ method to dynamically handle calls to methods that are not explicitly defined,
allowing it to handle arbitrary `on_<event_name>_start` and `on_<event_name>_end` methods.

Example:
    class MyPrintingCallback(Callback):
        def on_train_start(self, trainer):
            print("Training is starting")
"""


from scipy import stats

from sportslabkit.logger import logger
from sportslabkit.mot.base import Callback, MultiObjectTracker
from sportslabkit.types import Vector
from sportslabkit.vector_model import BaseVectorModel


class TeamClassificationCallback(Callback):
