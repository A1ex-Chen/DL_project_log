def log_extra_train_info(self, log_data):
    log_data.update({'cur_beta': self.__beta})
