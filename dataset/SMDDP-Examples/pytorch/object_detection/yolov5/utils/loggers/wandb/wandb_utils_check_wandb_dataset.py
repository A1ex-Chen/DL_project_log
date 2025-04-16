def check_wandb_dataset(data_file):
    is_trainset_wandb_artifact = False
    is_valset_wandb_artifact = False
    if isinstance(data_file, dict):
        return data_file
    if check_file(data_file) and data_file.endswith('.yaml'):
        with open(data_file, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        is_trainset_wandb_artifact = isinstance(data_dict['train'], str
            ) and data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX)
        is_valset_wandb_artifact = isinstance(data_dict['val'], str
            ) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX)
    if is_trainset_wandb_artifact or is_valset_wandb_artifact:
        return data_dict
    else:
        return check_dataset(data_file)
