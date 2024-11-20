def load(self, filename):
    """
        Loads checkpoint from filename.

        :param filename: path to the checkpoint file
        """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location={'cuda:0': 'cpu'})
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.fp_optimizer.initialize_model(self.model)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        logging.info(f'Loaded checkpoint {filename} (epoch {self.epoch})')
    else:
        logging.error(f'Invalid checkpoint: {filename}')
