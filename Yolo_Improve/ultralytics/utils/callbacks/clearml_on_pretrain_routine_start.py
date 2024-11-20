def on_pretrain_routine_start(trainer):
    """Runs at start of pretraining routine; initializes and connects/ logs task to ClearML."""
    try:
        if (task := Task.current_task()):
            from clearml.binding.frameworks.pytorch_bind import PatchPyTorchModelIO
            from clearml.binding.matplotlib_bind import PatchedMatplotlib
            PatchPyTorchModelIO.update_current_task(None)
            PatchedMatplotlib.update_current_task(None)
        else:
            task = Task.init(project_name=trainer.args.project or 'YOLOv8',
                task_name=trainer.args.name, tags=['YOLOv8'], output_uri=
                True, reuse_last_task_id=False, auto_connect_frameworks={
                'pytorch': False, 'matplotlib': False})
            LOGGER.warning(
                'ClearML Initialized a new task. If you want to run remotely, please add clearml-init and connect your arguments before initializing YOLO.'
                )
        task.connect(vars(trainer.args), name='General')
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ ClearML installed but not initialized correctly, not logging this run. {e}'
            )
