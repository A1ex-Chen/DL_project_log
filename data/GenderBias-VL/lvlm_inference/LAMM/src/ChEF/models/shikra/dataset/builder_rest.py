from functools import partial
from typing import Callable, Dict, Tuple, Any, Optional

from torch.utils.data import Dataset
from transformers import EvalPrediction, TrainingArguments

from .root import DATASETS, METRICS, TRANSFORMS, FUNCTIONS
from .single_image_convsation import SingleImageConvDataset
from .single_image_interactive import SingleImageInteractive
from ..conversation import get_conv_template
from .utils import init_ceph_client_if_needed

DatasetDict = Dict[str, Dataset]
ComputeMetrics = Callable[[EvalPrediction], Dict]





