def log_train(self, log_data):
    for logger in self.train_loggers:
        logger.log(**log_data)
