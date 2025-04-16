import logging 
import os
import sys
import json
import platform

from deepview_profile.analysis.runner import analyze_project
from deepview_profile.nvml import NVML
from deepview_profile.utils import release_memory, next_message_to_dict, files_encoded_unique

from deepview_profile.initialization import (
    check_skyline_preconditions,
    initialize_skyline,
)
from deepview_profile.error_printing import print_analysis_error

logger = logging.getLogger(__name__)









