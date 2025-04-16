def log_dataset_artifact(self, data_file, single_cls, project,
    overwrite_config=False):
    with open(data_file) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    nc, names = (1, ['item']) if single_cls else (int(data['nc']), data[
        'names'])
    names = {k: v for k, v in enumerate(names)}
    self.train_artifact = self.create_dataset_table(LoadImagesAndLabels(
        data['train']), names, name='train') if data.get('train') else None
    self.val_artifact = self.create_dataset_table(LoadImagesAndLabels(data[
        'val']), names, name='val') if data.get('val') else None
    if data.get('train'):
        data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')
    if data.get('val'):
        data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')
    path = data_file if overwrite_config else '_wandb.'.join(data_file.
        rsplit('.', 1))
    data.pop('download', None)
    with open(path, 'w') as f:
        yaml.dump(data, f)
    if self.job_type == 'Training':
        self.wandb_run.use_artifact(self.val_artifact)
        self.wandb_run.use_artifact(self.train_artifact)
        self.val_artifact.wait()
        self.val_table = self.val_artifact.get('val')
        self.map_val_table_path()
    else:
        self.wandb_run.log_artifact(self.train_artifact)
        self.wandb_run.log_artifact(self.val_artifact)
    return path
