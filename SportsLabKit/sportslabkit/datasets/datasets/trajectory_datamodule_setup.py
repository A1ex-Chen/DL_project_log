def setup(self, stage: str='both'):
    data_dir = self.data_dir
    train_data_dir = data_dir / 'train'
    val_data_dir = data_dir / 'val'
    test_data_dir = data_dir / 'test'
    if stage == 'fit' or stage == 'both':
        self.trainset = TrajectoryDataset(train_data_dir, transform=self.
            transform, split=self.split)
        self.valset = TrajectoryDataset(val_data_dir, transform=self.
            transform, split=self.split)
        self.testset = TrajectoryDataset(test_data_dir, transform=self.
            transform, split=self.split)
    if stage == 'test' or stage == 'both':
        self.testset = TrajectoryDataset(test_data_dir, transform=self.
            transform, split=self.split)
