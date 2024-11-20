def __init__(self, opt, run_id=None, job_type='Training'):
    """
        - Initialize WandbLogger instance
        - Upload dataset if opt.upload_dataset is True
        - Setup training processes if job_type is 'Training'

        arguments:
        opt (namespace) -- Commandline arguments for this run
        run_id (str) -- Run ID of W&B run to be resumed
        job_type (str) -- To set the job_type for this run

        """
    self.job_type = job_type
    self.wandb, self.wandb_run = wandb, wandb.run if wandb else None
    self.val_artifact, self.train_artifact = None, None
    self.train_artifact_path, self.val_artifact_path = None, None
    self.result_artifact = None
    self.val_table, self.result_table = None, None
    self.max_imgs_to_log = 16
    self.data_dict = None
    if self.wandb:
        self.wandb_run = wandb.run or wandb.init(config=opt, resume='allow',
            project='YOLOv3' if opt.project == 'runs/train' else Path(opt.
            project).stem, entity=opt.entity, name=opt.name if opt.name !=
            'exp' else None, job_type=job_type, id=run_id, allow_val_change
            =True)
    if self.wandb_run and self.job_type == 'Training':
        if isinstance(opt.data, dict):
            self.data_dict = opt.data
        self.setup_training(opt)
