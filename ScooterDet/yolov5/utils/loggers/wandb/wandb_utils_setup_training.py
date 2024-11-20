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
        model_dir, _ = self.download_model_artifact(opt)
        if model_dir:
            self.weights = Path(model_dir) / 'last.pt'
            config = self.wandb_run.config
            (opt.weights, opt.save_period, opt.batch_size, opt.
                bbox_interval, opt.epochs, opt.hyp, opt.imgsz) = (str(self.
                weights), config.save_period, config.batch_size, config.
                bbox_interval, config.epochs, config.hyp, config.imgsz)
    if opt.bbox_interval == -1:
        self.bbox_interval = opt.bbox_interval = (opt.epochs // 10 if opt.
            epochs > 10 else 1)
        if opt.evolve or opt.noplots:
            self.bbox_interval = opt.bbox_interval = opt.epochs + 1
