# coding=utf-8
# Copyright 2024 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logging utilities."""

import logging
import os
import sys
import threading
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from typing import Dict, Optional

from tqdm import auto as tqdm_lib


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING

_tqdm_active = True














































logging.Logger.warning_advice = warning_advice


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()












class _tqdm_cls:



        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    return bool(_tqdm_active)


def enable_progress_bar() -> None:
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True


def disable_progress_bar() -> None:
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False