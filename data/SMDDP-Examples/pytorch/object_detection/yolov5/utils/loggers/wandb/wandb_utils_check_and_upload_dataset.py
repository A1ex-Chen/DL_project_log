def check_and_upload_dataset(self, opt):
    """
        Check if the dataset format is compatible and upload it as W&B artifact

        arguments:
        opt (namespace)-- Commandline arguments for current run

        returns:
        Updated dataset info dictionary where local dataset paths are replaced by WAND_ARFACT_PREFIX links.
        """
    assert wandb, 'Install wandb to upload dataset'
    config_path = self.log_dataset_artifact(opt.data, opt.single_cls, 
        'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem)
    with open(config_path, errors='ignore') as f:
        wandb_data_dict = yaml.safe_load(f)
    return wandb_data_dict
