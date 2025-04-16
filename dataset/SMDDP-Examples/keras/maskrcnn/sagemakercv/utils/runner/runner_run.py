def run(self, dataset):
    if isinstance(dataset, dataset_ops._OptionsDataset):
        dataset = iter(dataset.repeat())
    if self.rank == 0:
        self.logger.info('Start running, work_dir: %s', self.work_dir)
        self.logger.info('max: %d epochs', self.max_epochs)
    self.call_hook('before_run')
    while self._epoch < self.max_epochs:
        self.train_epoch(dataset)
    self.call_hook('after_run')
