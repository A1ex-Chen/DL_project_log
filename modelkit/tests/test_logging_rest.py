import contextlib
import os

import structlog
from structlog.testing import capture_logs

from modelkit.utils.logging import ContextualizedLogging

test_path = os.path.dirname(os.path.realpath(__file__))




CONTEXT_RES = [
    {
        "context0": "value0",
        "event": "context0 message",
        "log_level": "info",
        "context": "value",
    },
    {
        "context0": "value0override",
        "event": "override context0 message",
        "log_level": "info",
        "context": "value",
    },
    {
        "context0": "value0",
        "context1": "value1",
        "extra_value": 1,
        "event": "context1 message",
        "log_level": "info",
        "context": "value",
    },
    {
        "context0": "value0",
        "event": "context0 message2",
        "log_level": "info",
        "context": "value",
    },
]


@contextlib.contextmanager

