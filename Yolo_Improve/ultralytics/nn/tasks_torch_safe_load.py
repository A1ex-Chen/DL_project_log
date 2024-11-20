def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset
    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)
    try:
        with temporary_modules(modules={'ultralytics.yolo.utils':
            'ultralytics.utils', 'ultralytics.yolo.v8':
            'ultralytics.models.yolo', 'ultralytics.yolo.data':
            'ultralytics.data'}, attributes={
            'ultralytics.nn.modules.block.Silence': 'torch.nn.Identity',
            'ultralytics.nn.tasks.YOLOv10DetectionModel':
            'ultralytics.nn.tasks.DetectionModel'}):
            ckpt = torch.load(file, map_location='cpu')
    except ModuleNotFoundError as e:
        if e.name == 'models':
            raise TypeError(emojis(
                f"""ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained with https://github.com/ultralytics/yolov5.
This model is NOT forwards compatible with YOLOv8 at https://github.com/ultralytics/ultralytics.
Recommend fixes are to train a new model using the latest 'ultralytics' package or to run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"""
                )) from e
        LOGGER.warning(
            f"""WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements.
AutoInstall will run now for '{e.name}' but this feature will be removed in the future.
Recommend fixes are to train a new model using the latest 'ultralytics' package or to run a command with an official Ultralytics model, i.e. 'yolo predict model=yolov8n.pt'"""
            )
        check_requirements(e.name)
        ckpt = torch.load(file, map_location='cpu')
    if not isinstance(ckpt, dict):
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. For optimal results, use model.save('filename.pt') to correctly save YOLO models."
            )
        ckpt = {'model': ckpt.model}
    return ckpt, file
