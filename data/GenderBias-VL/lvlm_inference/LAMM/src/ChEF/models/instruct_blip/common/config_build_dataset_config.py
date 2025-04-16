@staticmethod
def build_dataset_config(config):
    datasets = config.get('datasets', None)
    if datasets is None:
        raise KeyError(
            "Expecting 'datasets' as the root key for dataset configuration.")
    dataset_config = OmegaConf.create()
    for dataset_name in datasets:
        builder_cls = registry.get_builder_class(dataset_name)
        dataset_config_type = datasets[dataset_name].get('type', 'default')
        dataset_config_path = builder_cls.default_config_path(type=
            dataset_config_type)
        dataset_config = OmegaConf.merge(dataset_config, OmegaConf.load(
            dataset_config_path), {'datasets': {dataset_name: config[
            'datasets'][dataset_name]}})
    return dataset_config
