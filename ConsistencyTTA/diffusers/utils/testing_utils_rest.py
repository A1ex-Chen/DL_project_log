import inspect
import logging
import os
import random
import re
import tempfile
import unittest
import urllib.parse
from distutils.util import strtobool
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
from packaging import version

from .import_utils import (
    BACKENDS_MAPPING,
    is_compel_available,
    is_flax_available,
    is_note_seq_available,
    is_onnx_available,
    is_opencv_available,
    is_torch_available,
    is_torch_version,
)
from .logging import get_logger


global_rng = random.Random()

logger = get_logger(__name__)

if is_torch_available():
    import torch

    if "DIFFUSERS_TEST_DEVICE" in os.environ:
        torch_device = os.environ["DIFFUSERS_TEST_DEVICE"]

        available_backends = ["cuda", "cpu", "mps"]
        if torch_device not in available_backends:
            raise ValueError(
                f"unknown torch backend for diffusers tests: {torch_device}. Available backends are:"
                f" {available_backends}"
            )
        logger.info(f"torch_device overrode to {torch_device}")
    else:
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        is_torch_higher_equal_than_1_12 = version.parse(
            version.parse(torch.__version__).base_version
        ) >= version.parse("1.12")

        if is_torch_higher_equal_than_1_12:
            # Some builds of torch 1.12 don't have the mps backend registered. See #892 for more details
            mps_backend_registered = hasattr(torch.backends, "mps")
            torch_device = "mps" if (mps_backend_registered and torch.backends.mps.is_available()) else torch_device










_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_nightly_tests = parse_flag_from_env("RUN_NIGHTLY", default=False)


































# --- pytest conf functions --- #

# to avoid multiple invocation from tests/conftest.py and examples/conftest.py - make sure it's called only once
pytest_opt_registered = {}




            # note: not printing out any rep.sections to keep the report short

    # use ready-made report funcs, we are just hijacking the filehandle to log to a dedicated file each
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # note: some pytest plugins may interfere by hijacking the default `terminalreporter` (e.g.
    # pytest-instafail does that)

    # report failures with line/short/long styles
    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # short tb
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())
    with open(report_files["passes"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


class CaptureLogger:
    """
    Args:
    Context manager to capture `logging` streams
        logger: 'logging` logger object
    Returns:
        The captured output is available via `self.out`
    Example:
    ```python
    >>> from diffusers import logging
    >>> from diffusers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.py")
    >>> with CaptureLogger(logger) as cl:
    ...     logger.info(msg)
    >>> assert cl.out, msg + "\n"
    ```
    """



