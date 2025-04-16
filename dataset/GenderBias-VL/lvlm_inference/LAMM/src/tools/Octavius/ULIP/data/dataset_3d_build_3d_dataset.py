def build_3d_dataset(self, args, config):
    config = cfg_from_yaml_file(config)
    config.tokenizer = self.tokenizer
    config.train_transform = self.train_transform
    config.pretrain_dataset_prompt = self.pretrain_dataset_prompt
    config.validate_dataset_prompt = self.validate_dataset_prompt
    config.args = args
    config.use_height = args.use_height
    config.npoints = args.npoints
    config_others = EasyDict({'subset': self.dataset_split, 'whole': True})
    self.dataset = build_dataset_from_cfg(config, config_others)
