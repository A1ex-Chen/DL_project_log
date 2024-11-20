def on_pretrain_routine_start(trainer):
    """Initializes DVCLive logger for training metadata during pre-training routine."""
    try:
        global live
        live = dvclive.Live(save_dvc_exp=True, cache_images=True)
        LOGGER.info(
            "DVCLive is detected and auto logging is enabled (run 'yolo settings dvc=False' to disable)."
            )
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ DVCLive installed but not initialized correctly, not logging this run. {e}'
            )
