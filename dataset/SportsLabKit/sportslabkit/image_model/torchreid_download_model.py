def download_model(model_name):
    if model_name not in model_dict:
        raise ValueError(
            f'Model {model_name} not available. Available models are: {show_torchreid_models()}'
            )
    url = model_dict[model_name]
    filename = model_name + '.pth'
    file_path = model_save_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        logger.debug(f'Model {model_name} already exists in {model_save_dir}.')
        return file_path
    download_file_from_google_drive(url.split('/')[-2], file_path)
    logger.debug(
        f'Model {model_name} successfully downloaded and saved to {model_save_dir}.'
        )
    return file_path
