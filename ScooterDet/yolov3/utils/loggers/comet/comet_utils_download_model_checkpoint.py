def download_model_checkpoint(opt, experiment):
    model_dir = f'{opt.project}/{experiment.name}'
    os.makedirs(model_dir, exist_ok=True)
    model_name = COMET_MODEL_NAME
    model_asset_list = experiment.get_model_asset_list(model_name)
    if len(model_asset_list) == 0:
        logger.error(
            f'COMET ERROR: No checkpoints found for model name : {model_name}')
        return
    model_asset_list = sorted(model_asset_list, key=lambda x: x['step'],
        reverse=True)
    logged_checkpoint_map = {asset['fileName']: asset['assetId'] for asset in
        model_asset_list}
    resource_url = urlparse(opt.weights)
    checkpoint_filename = resource_url.query
    if checkpoint_filename:
        asset_id = logged_checkpoint_map.get(checkpoint_filename)
    else:
        asset_id = logged_checkpoint_map.get(COMET_DEFAULT_CHECKPOINT_FILENAME)
        checkpoint_filename = COMET_DEFAULT_CHECKPOINT_FILENAME
    if asset_id is None:
        logger.error(
            f'COMET ERROR: Checkpoint {checkpoint_filename} not found in the given Experiment'
            )
        return
    try:
        logger.info(f'COMET INFO: Downloading checkpoint {checkpoint_filename}'
            )
        asset_filename = checkpoint_filename
        model_binary = experiment.get_asset(asset_id, return_type='binary',
            stream=False)
        model_download_path = f'{model_dir}/{asset_filename}'
        with open(model_download_path, 'wb') as f:
            f.write(model_binary)
        opt.weights = model_download_path
    except Exception as e:
        logger.warning(
            'COMET WARNING: Unable to download checkpoint from Comet')
        logger.exception(e)
