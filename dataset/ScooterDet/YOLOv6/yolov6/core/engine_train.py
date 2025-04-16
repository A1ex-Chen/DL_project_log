def train(self):
    try:
        self.before_train_loop()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_one_epoch(self.epoch)
            self.after_epoch()
        self.strip_model()
    except Exception as _:
        LOGGER.error('ERROR in training loop or eval/save model.')
        raise
    finally:
        self.train_after_loop()
