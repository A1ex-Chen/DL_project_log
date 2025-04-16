import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import Dict, Generator, List, Set, Union

from tabulate import tabulate

from modelkit.core.model import Model
from modelkit.core.profilers.base import BaseProfiler


class SimpleProfiler(BaseProfiler):
    """This simple profiler records the duration of model prediction in seconds,
    and compute the net percentage duration of each sub models via 'model_dependencies'.
    Usage:
        model = modelkit.load_model(...)
        profiler = SimpleProfiler(model)
        res = model(item)
        profiler.summary() # return profiling result (Dict) or str

    Attributes:
        recording_hook (Dict[str, float]): record start/end time of each model call
        durations (List[float]): record duration of each model call
        net_durations (List[float]): record net duration of each model call. Net
            duration is the duration minus all the other sub models' duration.
        graph (Dict[str, Set]): model dependencies graph, get all direct children names
            (Set[str])
        graph_calls (Dict[str, Dict[str, int]]): record all model calls
            e.g
            {
                "pipeline": {
                    "__main__": 1, # "pipeline" is called once
                    "model_a": 2,
                    "model_b": 1,
                    "model_c": 1,
                }
            }
    See test_simple_profiler.py for more details.
    """




    @contextmanager




