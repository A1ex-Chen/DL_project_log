# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, checks

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["dvc"] is True  # verify integration is enabled
    import dvclive

    assert checks.check_version("dvclive", "2.11.0", verbose=True)

    import os
    import re
    from pathlib import Path

    # DVCLive logger instance
    live = None
    _processed_plots = {}

    # `on_fit_epoch_end` is called on final validation (probably need to be fixed) for now this is the way we
    # distinguish final evaluation of the best model vs last epoch validation
    _training_epoch = False

except (ImportError, AssertionError, TypeError):
    dvclive = None




















callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if dvclive
    else {}
)