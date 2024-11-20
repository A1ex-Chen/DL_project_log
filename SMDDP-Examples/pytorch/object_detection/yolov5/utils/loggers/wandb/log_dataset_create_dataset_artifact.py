def create_dataset_artifact(opt):
    logger = WandbLogger(opt, None, job_type='Dataset Creation')
    if not logger.wandb:
        LOGGER.info(
            'install wandb using `pip install wandb` to log the dataset')
