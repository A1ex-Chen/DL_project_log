def check_dataset(self, data_file):
    with open(data_file) as f:
        data_config = yaml.safe_load(f)
    path = data_config.get('path')
    if path and path.startswith(COMET_PREFIX):
        path = data_config['path'].replace(COMET_PREFIX, '')
        data_dict = self.download_dataset_artifact(path)
        return data_dict
    self.log_asset(self.opt.data, metadata={'type': 'data-config-file'})
    return check_dataset(data_file)
