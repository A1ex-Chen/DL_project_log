def train_epoch(self, dataset, steps=None, **kwargs):
    if self.rank == 0:
        self.logger.info(
            f'Starting epoch: {self._epoch + 1} of {self.max_epochs}')
    if isinstance(dataset, dataset_ops._OptionsDataset):
        dataset = iter(dataset.repeat())
    if steps == None:
        steps = self.steps_per_epoch
    self.num_examples = steps
    self.mode = 'train'
    broadcast = True
    self.call_hook('before_train_epoch')
    self._inner_iter = 0
    for i in range(self.num_examples):
        self.data_batch = next(dataset)
        self.call_hook('before_train_iter')
        self.losses, self.outputs = self.trainer(self.data_batch, training=
            True, broadcast=broadcast)
        broadcast = False
        if not isinstance(self.losses, dict):
            raise TypeError('trainer must return a dict')
        if self.rank == 0:
            self.log_buffer.update(self.losses, self._iter)
        self.call_hook('after_train_iter')
        self._iter += 1
        self._inner_iter += 1
        if self._iter >= self.max_iters:
            break
    self._epoch += 1
    self.call_hook('after_train_epoch')
