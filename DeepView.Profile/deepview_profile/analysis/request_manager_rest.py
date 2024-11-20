import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from deepview_profile.analysis.runner import analyze_project
from deepview_profile.exceptions import AnalysisError
from deepview_profile.nvml import NVML
import deepview_profile.protocol_gen.innpv_pb2 as pm
import sys
logger = logging.getLogger(__name__)


class AnalysisRequestManager:












