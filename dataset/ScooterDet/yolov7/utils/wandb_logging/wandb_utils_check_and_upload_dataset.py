def check_and_upload_dataset(self, opt):
    assert wandb, 'Install wandb to upload dataset'
    check_dataset(self.data_dict)
    config_path = self.log_dataset_artifact(opt.data, opt.single_cls, 
        'YOLOR' if opt.project == 'runs/train' else Path(opt.project).stem)
    print('Created dataset config file ', config_path)
    with open(config_path) as f:
        wandb_data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    return wandb_data_dict
