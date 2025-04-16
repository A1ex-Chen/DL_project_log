def __init__(self, opt, name, run_id, data_dict, job_type='Training'):
    self.job_type = job_type
    self.wandb, self.wandb_run, self.data_dict = (wandb, None if not wandb else
        wandb.run, data_dict)
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            run_id, project, model_artifact_name = get_run_info(opt.resume)
            model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
            assert wandb, 'install wandb to resume wandb runs'
            self.wandb_run = wandb.init(id=run_id, project=project, resume=
                'allow')
            opt.resume = model_artifact_name
    elif self.wandb:
        self.wandb_run = wandb.init(config=opt, resume='allow', project=
            'YOLOR' if opt.project == 'runs/train' else Path(opt.project).
            stem, name=name, job_type=job_type, id=run_id
            ) if not wandb.run else wandb.run
    if self.wandb_run:
        if self.job_type == 'Training':
            if not opt.resume:
                wandb_data_dict = self.check_and_upload_dataset(opt
                    ) if opt.upload_dataset else data_dict
                self.wandb_run.config.opt = vars(opt)
                self.wandb_run.config.data_dict = wandb_data_dict
            self.data_dict = self.setup_training(opt, data_dict)
        if self.job_type == 'Dataset Creation':
            self.data_dict = self.check_and_upload_dataset(opt)
    else:
        prefix = colorstr('wandb: ')
        print(
            f"{prefix}Install Weights & Biases for YOLOR logging with 'pip install wandb' (recommended)"
            )
