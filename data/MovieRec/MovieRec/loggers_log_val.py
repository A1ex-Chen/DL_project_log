def log_val(self, log_data):
    for logger in self.val_loggers:
        logger.log(**log_data)
