def check_wandb_config_file(data_config_file):
    wandb_config = '_wandb.'.join(data_config_file.rsplit('.', 1))
    if Path(wandb_config).is_file():
        return wandb_config
    return data_config_file
