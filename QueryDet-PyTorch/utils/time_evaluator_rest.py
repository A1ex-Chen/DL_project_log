import time
from detectron2.evaluation.evaluator import DatasetEvaluator
import detectron2.utils.comm as comm
import itertools
from collections import OrderedDict 

import numpy as np 

class GPUTimeEvaluator(DatasetEvaluator):
    

