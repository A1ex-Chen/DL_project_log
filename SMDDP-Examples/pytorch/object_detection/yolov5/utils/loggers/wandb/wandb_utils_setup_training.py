def setup_training(self, opt):
    """
        Setup the necessary processes for training YOLO models:
          - Attempt to download model checkpoint and dataset artifacts if opt.resume stats with WANDB_ARTIFACT_PREFIX
          - Update data_dict, to contain info of previous run if resumed and the paths of dataset artifact if downloaded
          - Setup log_dict, initialize bbox_interval

        arguments:
        opt (namespace) -- commandline arguments for this run

        """
    self.log_dict, self.current_epoch = {}, 0
    self.bbox_interval = opt.bbox_interval
    if isinstance(opt.resume, str):
        modeldir, _ = self.download_model_artifact(opt)
        if modeldir:
            self.weights = Path(modeldir) / 'last.pt'
            config = self.wandb_run.config
            (opt.weights, opt.save_period, opt.batch_size, opt.
                bbox_interval, opt.epochs, opt.hyp, opt.imgsz) = (str(self.
                weights), config.save_period, config.batch_size, config.
                bbox_interval, config.epochs, config.hyp, config.imgsz)
    data_dict = self.data_dict
    if self.val_artifact is None:
        self.train_artifact_path, self.train_artifact = (self.
            download_dataset_artifact(data_dict.get('train'), opt.
            artifact_alias))
        self.val_artifact_path, self.val_artifact = (self.
            download_dataset_artifact(data_dict.get('val'), opt.artifact_alias)
            )
    if self.train_artifact_path is not None:
        train_path = Path(self.train_artifact_path) / 'data/images/'
        data_dict['train'] = str(train_path)
    if self.val_artifact_path is not None:
        val_path = Path(self.val_artifact_path) / 'data/images/'
        data_dict['val'] = str(val_path)
    if self.val_artifact is not None:
        self.result_artifact = wandb.Artifact('run_' + wandb.run.id +
            '_progress', 'evaluation')
        columns = ['epoch', 'id', 'ground truth', 'prediction']
        columns.extend(self.data_dict['names'])
        self.result_table = wandb.Table(columns)
        self.val_table = self.val_artifact.get('val')
        if self.val_table_path_map is None:
            self.map_val_table_path()
    if opt.bbox_interval == -1:
        self.bbox_interval = opt.bbox_interval = (opt.epochs // 10 if opt.
            epochs > 10 else 1)
        if opt.evolve or opt.noplots:
            self.bbox_interval = opt.bbox_interval = opt.epochs + 1
    train_from_artifact = (self.train_artifact_path is not None and self.
        val_artifact_path is not None)
    if train_from_artifact:
        self.data_dict = data_dict
