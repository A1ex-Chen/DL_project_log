def __init__(self, path='coco128.yaml', autodownload=False):
    zipped, data_dir, yaml_path = self._unzip(Path(path))
    try:
        with open(check_yaml(yaml_path), errors='ignore') as f:
            data = yaml.safe_load(f)
            if zipped:
                data['path'] = data_dir
    except Exception as e:
        raise Exception('error/HUB/dataset_stats/yaml_load') from e
    check_dataset(data, autodownload)
    self.hub_dir = Path(data['path'] + '-hub')
    self.im_dir = self.hub_dir / 'images'
    self.im_dir.mkdir(parents=True, exist_ok=True)
    self.stats = {'nc': data['nc'], 'names': data['names']}
    self.data = data
