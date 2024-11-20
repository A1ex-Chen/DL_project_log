import random
import unittest

import numpy as np
import pandas as pd

import soccertrack
from soccertrack.dataframe.bboxdataframe import BBoxDataFrame
from soccertrack.metrics import (
    ap_score,
    ap_score_range,
    hota_score,
    identity_score,
    iou_score,
    map_score,
    map_score_range,
    mota_score,
)
from soccertrack.metrics.tracking_preprocess import to_mot_eval_format

# global variables for tracking evaluation
dataset_path = soccertrack.datasets.get_path("top_view")
path_to_csv = sorted(dataset_path.glob("annotations/*.csv"))[0]
bbdf = soccertrack.load_df(path_to_csv)[0:2]
player_dfs = [player_df for _, player_df in bbdf.iter_players(drop=False)]


class TestMetrics(unittest.TestCase):























if __name__ == "__main__":
    unittest.main()