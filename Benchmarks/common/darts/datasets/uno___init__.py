def __init__(self, root, partition, transform=None, target_transform=None,
    download=False):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    if download:
        self.download()
    if not self._check_exists():
        raise RuntimeError('Dataset not found.' +
            ' You can use download=True to download it')
    self.partition = partition
    if self.partition == 'train':
        data_file = self.training_data_file
        label_file = self.training_label_file
    elif self.partition == 'test':
        data_file = self.test_data_file
        label_file = self.test_label_file
    else:
        raise ValueError("Partition must either be 'train' or 'test'.")
    self.data = torch.load(os.path.join(self.processed_folder, data_file))
    self.targets = torch.load(os.path.join(self.processed_folder, label_file))
