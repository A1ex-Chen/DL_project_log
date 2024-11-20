import contextlib
import torch

from deepview_profile.exceptions import _SuspendExecution
from deepview_profile.tracking.hook_manager import HookManager


class BackwardInterceptor:

    @contextlib.contextmanager

        return hook