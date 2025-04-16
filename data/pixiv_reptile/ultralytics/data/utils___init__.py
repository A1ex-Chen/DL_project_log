def __init__(self, path='coco8.yaml', task='detect', autodownload=False):
    """Initialize class."""
    path = Path(path).resolve()
    LOGGER.info(f'Starting HUB dataset checks for {path}....')
    self.task = task
    if self.task == 'classify':
        unzip_dir = unzip_file(path)
        data = check_cls_dataset(unzip_dir)
        data['path'] = unzip_dir
    else:
        _, data_dir, yaml_path = self._unzip(Path(path))
        try:
            data = yaml_load(yaml_path)
            data['path'] = ''
            yaml_save(yaml_path, data)
            data = check_det_dataset(yaml_path, autodownload)
            data['path'] = data_dir
        except Exception as e:
            raise Exception('error/HUB/dataset_stats/init') from e
    self.hub_dir = Path(f"{data['path']}-hub")
    self.im_dir = self.hub_dir / 'images'
    self.stats = {'nc': len(data['names']), 'names': list(data['names'].
        values())}
    self.data = data
