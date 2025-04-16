"""Main Logger class for ClearML experiment tracking."""
import glob
import re
from pathlib import Path

import yaml
from torchvision.transforms import ToPILImage

try:
    import clearml
    from clearml import Dataset, Task
    from torchvision.utils import draw_bounding_boxes  # WARNING: requires torchvision>=0.9.0

    assert hasattr(clearml, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    clearml = None




class ClearmlLogger:
    """Log training runs, datasets, models, and predictions to ClearML.

    This logger sends information to ClearML at app.clear.ml or to your own hosted server. By default,
    this information includes hyperparameters, system configuration and metrics, model metrics, code information and
    basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.
    """


