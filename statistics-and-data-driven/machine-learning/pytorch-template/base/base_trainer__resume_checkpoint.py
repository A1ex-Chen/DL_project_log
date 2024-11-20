def _resume_checkpoint(self, resume_path):
    """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
    resume_path = str(resume_path)
    self.logger.info('Loading checkpoint: {} ...'.format(resume_path))
    checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch'] + 1
    self.mnt_best = checkpoint['monitor_best']
    if checkpoint['config']['arch'] != self.config['arch']:
        self.logger.warning(
            'Warning: Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded.'
            )
    self.model.load_state_dict(checkpoint['state_dict'])
    if checkpoint['config']['optimizer']['type'] != self.config['optimizer'][
        'type']:
        self.logger.warning(
            'Warning: Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed.'
            )
    else:
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.logger.info('Checkpoint loaded. Resume training from epoch {}'.
        format(self.start_epoch))
