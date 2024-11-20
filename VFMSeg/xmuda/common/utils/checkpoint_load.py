def load(self, path=None, resume=True, resume_states=True):
    if resume and self.has_checkpoint():
        path = self.get_checkpoint_file()
    if not path:
        self._print('No checkpoint found. Initializing model from scratch')
        return {}
    self._print('Loading checkpoint from {}, MD5: {}'.format(path, get_md5(
        path)))
    checkpoint = self._load_file(path)
    if isinstance(self.model, (DataParallel, DistributedDataParallel)):
        self.model.module.load_state_dict(checkpoint.pop('model'))
    else:
        self.model.load_state_dict(checkpoint.pop('model'))
    if resume_states:
        if 'optimizer' in checkpoint and self.optimizer:
            self.logger.info('Loading optimizer from {}'.format(path))
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if 'scheduler' in checkpoint and self.scheduler:
            self.logger.info('Loading scheduler from {}'.format(path))
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
    else:
        checkpoint = {}
    return checkpoint
