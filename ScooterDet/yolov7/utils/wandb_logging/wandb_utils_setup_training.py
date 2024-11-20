def setup_training(self, opt, data_dict):
    self.log_dict, self.current_epoch, self.log_imgs = {}, 0, 16
    self.bbox_interval = opt.bbox_interval
    if isinstance(opt.resume, str):
        modeldir, _ = self.download_model_artifact(opt)
        if modeldir:
            self.weights = Path(modeldir) / 'last.pt'
            config = self.wandb_run.config
            (opt.weights, opt.save_period, opt.batch_size, opt.
                bbox_interval, opt.epochs, opt.hyp) = (str(self.weights),
                config.save_period, config.total_batch_size, config.
                bbox_interval, config.epochs, config.opt['hyp'])
        data_dict = dict(self.wandb_run.config.data_dict)
    if 'val_artifact' not in self.__dict__:
        self.train_artifact_path, self.train_artifact = (self.
            download_dataset_artifact(data_dict.get('train'), opt.
            artifact_alias))
        self.val_artifact_path, self.val_artifact = (self.
            download_dataset_artifact(data_dict.get('val'), opt.artifact_alias)
            )
        (self.result_artifact, self.result_table, self.val_table, self.weights
            ) = None, None, None, None
        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path) / 'data/images/'
            data_dict['train'] = str(train_path)
        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path) / 'data/images/'
            data_dict['val'] = str(val_path)
            self.val_table = self.val_artifact.get('val')
            self.map_val_table_path()
    if self.val_artifact is not None:
        self.result_artifact = wandb.Artifact('run_' + wandb.run.id +
            '_progress', 'evaluation')
        self.result_table = wandb.Table(['epoch', 'id', 'prediction',
            'avg_confidence'])
    if opt.bbox_interval == -1:
        self.bbox_interval = opt.bbox_interval = (opt.epochs // 10 if opt.
            epochs > 10 else 1)
    return data_dict
