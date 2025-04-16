def process_wandb_config_ddp_mode(opt):
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    train_dir, val_dir = None, None
    if isinstance(data_dict['train'], str) and data_dict['train'].startswith(
        WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        train_artifact = api.artifact(remove_prefix(data_dict['train']) +
            ':' + opt.artifact_alias)
        train_dir = train_artifact.download()
        train_path = Path(train_dir) / 'data/images/'
        data_dict['train'] = str(train_path)
    if isinstance(data_dict['val'], str) and data_dict['val'].startswith(
        WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        val_artifact = api.artifact(remove_prefix(data_dict['val']) + ':' +
            opt.artifact_alias)
        val_dir = val_artifact.download()
        val_path = Path(val_dir) / 'data/images/'
        data_dict['val'] = str(val_path)
    if train_dir or val_dir:
        ddp_data_path = str(Path(val_dir) / 'wandb_local_data.yaml')
        with open(ddp_data_path, 'w') as f:
            yaml.dump(data_dict, f)
        opt.data = ddp_data_path
