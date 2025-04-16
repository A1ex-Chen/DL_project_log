from collections import defaultdict
import json
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List

from mypy_extensions import TypedDict


Prediction = TypedDict("Prediction", {"image_id": int, "caption": str})


class NocapsEvaluator(object):
