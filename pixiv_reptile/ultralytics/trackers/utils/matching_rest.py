# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lapx>=0.5.2")  # update to lap package from https://github.com/rathaROG/lapx
    import lap







