def __init__(self, args, tokenizer, dataset_type, train_transform=None):
    if dataset_type == 'train':
        self.dataset_name = args.pretrain_dataset_name
    elif dataset_type == 'val':
        self.dataset_name = args.validate_dataset_name
    else:
        raise ValueError('not supported dataset type.')
    with open('./data/dataset_catalog.json', 'r') as f:
        self.dataset_catalog = json.load(f)
        self.dataset_usage = self.dataset_catalog[self.dataset_name]['usage']
        self.dataset_split = self.dataset_catalog[self.dataset_name][self.
            dataset_usage]
        self.dataset_config_dir = self.dataset_catalog[self.dataset_name][
            'config']
    self.tokenizer = tokenizer
    self.train_transform = train_transform
    self.pretrain_dataset_prompt = args.pretrain_dataset_prompt
    self.validate_dataset_prompt = args.validate_dataset_prompt
    self.build_3d_dataset(args, self.dataset_config_dir)
