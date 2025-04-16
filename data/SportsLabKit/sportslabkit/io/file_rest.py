from __future__ import annotations

import csv
import os
from ast import literal_eval
from collections.abc import Callable, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any, Union

import dateutil.parser
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ..dataframe import BBoxDataFrame, CoordinatesDataFrame


PathLike = Union[str, os.PathLike]








































# def load_bboxes_from_yaml(yaml_path: PathLike) -> BBoxDataFrame:
#     """
#     Args:
#         yaml_path(str): Path to yaml file.

#     Returns:
#         merged_dataframe(BBoxDataFrame):
#     """

#     cfg = OmegaConf.load(yaml_path)
#     df_list = []
#     playerids, teamids, filepaths = [], [], []
#     for device in cfg.devices:
#         playerids.append(device.playerid)
#         teamids.append(device.teamid)
#         filepaths.append(Path(device.filepath))

#     return load_bboxes(filepaths, playerids, teamids)