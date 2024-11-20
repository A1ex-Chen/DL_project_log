# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["clearml"] is True  # verify integration is enabled
    import clearml
    from clearml import Task

    assert hasattr(clearml, "__version__")  # verify package is not directory

except (ImportError, AssertionError):
    clearml = None
















callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    if clearml
    else {}
)