import os
from functools import lru_cache
from logging import warn
from typing import Sequence, Tuple

import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table












class DummyOpenCVImport:
    def __getattribute__(self, name):
        print(
            r"""[bold red]Missing dependency:[/bold red] You are trying to use Norfair's video features. However, OpenCV is not installed.

Please, make sure there is an existing installation of OpenCV or install Norfair with `pip install norfair\[video]`."""
        )
        exit()


class DummyMOTMetricsImport:
    def __getattribute__(self, name):
        print(
            r"""[bold red]Missing dependency:[/bold red] You are trying to use Norfair's metrics features without the required dependencies.

Please, install Norfair with `pip install norfair\[metrics]`, or `pip install norfair\[metrics,video]` if you also want video features."""
        )
        exit()


# lru_cache will prevent re-run the function if the message is the same
@lru_cache(maxsize=None)


class DummyMOTMetricsImport:


# lru_cache will prevent re-run the function if the message is the same
@lru_cache(maxsize=None)
def warn_once(message):
    """
    Write a warning message only once.
    """
    warn(message)