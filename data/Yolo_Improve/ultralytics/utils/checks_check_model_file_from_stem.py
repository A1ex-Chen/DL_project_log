def check_model_file_from_stem(model='yolov8n'):
    """Return a model filename from a valid model stem."""
    if model and not Path(model).suffix and Path(model
        ).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix('.pt')
    else:
        return model
