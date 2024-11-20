import numpy as np
import onnx
from onnx import shape_inference
try:
    import onnx_graphsurgeon as gs
except Exception as e:
    print('Import onnx_graphsurgeon failure: %s' % e)

import logging

LOGGER = logging.getLogger(__name__)

class RegisterNMS(object):


