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
    self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run
    self.val_artifact, self.train_artifact = None, None
    self.train_artifact_path, self.val_artifact_path = None, None
    self.result_artifact = None
    self.val_table, self.result_table = None, None
    self.bbox_media_panel_images = []
    self.val_table_path_map = None
    self.max_imgs_to_log = 16
    self.wandb_artifact_data_dict = None
    self.data_dict = None
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            entity, project, run_id, model_artifact_name = get_run_info(opt
                .resume)
            model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
            assert wandb, 'install wandb to resume wandb runs'
            self.wandb_run = wandb.init(id=run_id, project=project, entity=
                entity, resume='allow', allow_val_change=True)
            opt.resume = model_artifact_name
    elif self.wandb:
        self.wandb_run = wandb.init(config=opt, resume='allow', project=
            'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).
            stem, entity=opt.entity, name=opt.name if opt.name != 'exp' else
            None, job_type=job_type, id=run_id, allow_val_change=True
            ) if not wandb.run else wandb.run
    if self.wandb_run:
        if self.job_type == 'Training':
            if opt.upload_dataset:
                if not opt.resume:
                    self.wandb_artifact_data_dict = (self.
                        check_and_upload_dataset(opt))
            if isinstance(opt.data, dict):
                self.data_dict = opt.data
            elif opt.resume:
                if isinstance(opt.resume, str) and opt.resume.startswith(
                    WANDB_ARTIFACT_PREFIX):
                    self.data_dict = dict(self.wandb_run.config.data_dict)
                else:
                    self.data_dict = check_wandb_dataset(opt.data)
            else:
                self.data_dict = check_wandb_dataset(opt.data)
                self.wandb_artifact_data_dict = (self.
                    wandb_artifact_data_dict or self.data_dict)
                self.wandb_run.config.update({'data_dict': self.
                    wandb_artifact_data_dict}, allow_val_change=True)
            self.setup_training(opt)
        if self.job_type == 'Dataset Creation':
            self.wandb_run.config.update({'upload_dataset': True})
            self.data_dict = self.check_and_upload_dataset(opt)
