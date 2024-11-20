from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from sportslabkit import BBoxDataFrame
from sportslabkit.metrics.object_detection import convert_to_x1y1x2y2, iou_score

