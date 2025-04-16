def setup(self, mode):
    if mode == 'test':
        self.dataset_size = len(self.test_dataloader().dataset)
    else:
        self.train_loader = self.get_dataloader('train', self.hparams.
            train_batch_size, shuffle=True)
        self.dataset_size = len(self.train_dataloader().dataset)
