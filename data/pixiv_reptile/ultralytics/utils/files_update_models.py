def update_models(model_names=('yolov8n.pt',), source_dir=Path(''),
    update_names=False):
    """
    Updates and re-saves specified YOLO models in an 'updated_models' subdirectory.

    Args:
        model_names (tuple, optional): Model filenames to update, defaults to ("yolov8n.pt").
        source_dir (Path, optional): Directory containing models and target subdirectory, defaults to current directory.
        update_names (bool, optional): Update model names from a data YAML.

    Example:
        ```python
        from ultralytics.utils.files import update_models

        model_names = (f"rtdetr-{size}.pt" for size in "lx")
        update_models(model_names)
        ```
    """
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names
    target_dir = source_dir / 'updated_models'
    target_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        model_path = source_dir / model_name
        print(f'Loading model from {model_path}')
        model = YOLO(model_path)
        model.half()
        if update_names:
            model.model.names = default_class_names('coco8.yaml')
        save_path = target_dir / model_name
        print(f'Re-saving {model_name} model to {save_path}')
        model.save(save_path, use_dill=False)
