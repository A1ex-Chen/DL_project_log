def __init__(self, file=SETTINGS_YAML, version='0.0.4'):
    """Initialize the SettingsManager with default settings, load and validate current settings from the YAML
        file.
        """
    import copy
    import hashlib
    from ultralytics.utils.checks import check_version
    from ultralytics.utils.torch_utils import torch_distributed_zero_first
    root = GIT_DIR or Path()
    datasets_root = (root.parent if GIT_DIR and is_dir_writeable(root.
        parent) else root).resolve()
    self.file = Path(file)
    self.version = version
    self.defaults = {'settings_version': version, 'datasets_dir': str(
        datasets_root / 'datasets'), 'weights_dir': str(root / 'weights'),
        'runs_dir': str(root / 'runs'), 'uuid': hashlib.sha256(str(uuid.
        getnode()).encode()).hexdigest(), 'sync': True, 'api_key': '',
        'openai_api_key': '', 'clearml': True, 'comet': True, 'dvc': True,
        'hub': True, 'mlflow': True, 'neptune': True, 'raytune': True,
        'tensorboard': True, 'wandb': True}
    super().__init__(copy.deepcopy(self.defaults))
    with torch_distributed_zero_first(RANK):
        if not self.file.exists():
            self.save()
        self.load()
        correct_keys = self.keys() == self.defaults.keys()
        correct_types = all(type(a) is type(b) for a, b in zip(self.values(
            ), self.defaults.values()))
        correct_version = check_version(self['settings_version'], self.version)
        help_msg = f"""
View settings with 'yolo settings' or at '{self.file}'
Update settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings."""
        if not (correct_keys and correct_types and correct_version):
            LOGGER.warning(
                f'WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem with your settings or a recent ultralytics package update. {help_msg}'
                )
            self.reset()
        if self.get('datasets_dir') == self.get('runs_dir'):
            LOGGER.warning(
                f"WARNING ⚠️ Ultralytics setting 'datasets_dir: {self.get('datasets_dir')}' must be different than 'runs_dir: {self.get('runs_dir')}'. Please change one to avoid possible issues during training. {help_msg}"
                )
