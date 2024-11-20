# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, ops

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["comet"] is True  # verify integration is enabled
    import comet_ml

    assert hasattr(comet_ml, "__version__")  # verify package is not directory

    import os
    from pathlib import Path

    # Ensures certain logging functions only run for supported tasks
    COMET_SUPPORTED_TASKS = ["detect"]

    # Names of plots created by YOLOv8 that are logged to Comet
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve", "confusion_matrix"
    LABEL_PLOT_NAMES = "labels", "labels_correlogram"

    _comet_image_prediction_count = 0

except (ImportError, AssertionError):
    comet_ml = None


















































callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if comet_ml
    else {}
)