def log_dataset_artifact(self, data_file, single_cls, project,
    overwrite_config=False):
    """
        Log the dataset as W&B artifact and return the new data file with W&B links

        arguments:
        data_file (str) -- the .yaml file with information about the dataset like - path, classes etc.
        single_class (boolean)  -- train multi-class data as single-class
        project (str) -- project name. Used to construct the artifact path
        overwrite_config (boolean) -- overwrites the data.yaml file if set to true otherwise creates a new
        file with _wandb postfix. Eg -> data_wandb.yaml

        returns:
        the new .yaml file with artifact links. it can be used to start training directly from artifacts
        """
    upload_dataset = self.wandb_run.config.upload_dataset
    log_val_only = isinstance(upload_dataset, str) and upload_dataset == 'val'
    self.data_dict = check_dataset(data_file)
    data = dict(self.data_dict)
    nc, names = (1, ['item']) if single_cls else (int(data['nc']), data[
        'names'])
    names = {k: v for k, v in enumerate(names)}
    if not log_val_only:
        self.train_artifact = self.create_dataset_table(LoadImagesAndLabels
            (data['train'], rect=True, batch_size=1), names, name='train'
            ) if data.get('train') else None
        if data.get('train'):
            data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train'
                )
    self.val_artifact = self.create_dataset_table(LoadImagesAndLabels(data[
        'val'], rect=True, batch_size=1), names, name='val') if data.get('val'
        ) else None
    if data.get('val'):
        data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
    path = Path(data_file)
    if not log_val_only:
        path = (path.stem if overwrite_config else path.stem + '_wandb'
            ) + '.yaml'
        path = ROOT / 'data' / path
        data.pop('download', None)
        data.pop('path', None)
        with open(path, 'w') as f:
            yaml.safe_dump(data, f)
            LOGGER.info(f'Created dataset config file {path}')
    if self.job_type == 'Training':
        if not log_val_only:
            self.wandb_run.log_artifact(self.train_artifact)
        self.wandb_run.use_artifact(self.val_artifact)
        self.val_artifact.wait()
        self.val_table = self.val_artifact.get('val')
        self.map_val_table_path()
    else:
        self.wandb_run.log_artifact(self.train_artifact)
        self.wandb_run.log_artifact(self.val_artifact)
    return path
