import sys
import logging
from typing import Dict, Any, Sequence

from transformers import EvalPrediction

from ...utils import decode_generate_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), ],
)


class BaseComputeMetrics:


