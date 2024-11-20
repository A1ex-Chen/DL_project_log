# Ultralytics YOLO 🚀, AGPL-3.0 license

import subprocess

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks



    # Get search space
    if not space:
        space = default_space
        LOGGER.warning("WARNING ⚠️ search space not provided, using default search space.")

    # Get dataset
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'WARNING ⚠️ data not provided, using default "data={data}".')

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    # Define the ASHA scheduler for hyperparameter search
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100,
        grace_period=grace_period,
        reduction_factor=3,
    )

    # Define the callbacks for the hyperparameter search
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv8-tune")] if wandb else []

    # Create the Ray Tune hyperparameter search tuner
    tune_dir = get_save_dir(DEFAULT_CFG, name="tune").resolve()  # must be absolute dir
    tune_dir.mkdir(parents=True, exist_ok=True)
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=space,
        tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
        run_config=RunConfig(callbacks=tuner_callbacks, storage_path=tune_dir),
    )

    # Run the hyperparameter search
    tuner.fit()

    # Return the results of the hyperparameter search
    return tuner.get_results()