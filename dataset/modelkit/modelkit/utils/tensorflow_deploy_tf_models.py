def deploy_tf_models(lib, mode, config_name='config', verbose=False):
    manager = AssetsManager()
    configuration = lib.configuration
    model_paths = {}
    if mode == 'remote':
        if not manager.storage_provider:
            raise ValueError(
                'A remote storage provider is required for `remote` mode')
        driver = manager.storage_provider.driver
    for model_name in lib.required_models:
        model_configuration = configuration[model_name]
        if not issubclass(model_configuration.model_type, TensorflowModel):
            logger.debug(f'Skipping non TF model `{model_name}`')
            continue
        if not model_configuration.asset:
            raise ValueError(
                f'TensorFlow model `{model_name}` does not have an asset')
        spec = AssetSpec.from_string(model_configuration.asset)
        if mode == 'local-docker':
            model_paths[model_name] = '/'.join(('/config', spec.name, spec.
                version or '')) + (spec.sub_part or '')
        elif mode == 'local-process':
            model_paths[model_name] = os.path.join(manager.assets_dir, *
                spec.name.split('/'), f'{spec.version}', *(spec.sub_part.
                split('/') if spec.sub_part else ()))
        elif mode == 'remote':
            object_name = manager.storage_provider.get_object_name(spec.
                name, spec.version or '')
            model_paths[model_name] = driver.get_object_uri(object_name)
    if mode == 'local-docker' or mode == 'local-process':
        logger.info('Checking that local models are present.')
        download_assets(configuration=configuration, required_models=lib.
            required_models)
    target = os.path.join(manager.assets_dir, f'{config_name}.config')
    if model_paths:
        logger.info('Writing TF serving configuration locally.',
            config_name=config_name, target=target)
        write_config(target, model_paths, verbose=verbose)
    else:
        logger.info('Nothing to write', config_name=config_name, target=target)
