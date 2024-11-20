def train_one_epoch(self, epoch_num):
    try:
        for self.step, self.batch_data in self.pbar:
            self.train_in_steps(epoch_num, self.step)
            self.print_details()
    except Exception as _:
        LOGGER.error('ERROR in training steps.')
        raise
