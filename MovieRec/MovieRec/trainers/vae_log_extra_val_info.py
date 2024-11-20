def log_extra_val_info(self, log_data):
    if self.finding_best_beta:
        log_data.update({'best_beta': self.best_beta})
