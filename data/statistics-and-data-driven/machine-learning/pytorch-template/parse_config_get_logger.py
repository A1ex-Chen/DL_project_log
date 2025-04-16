def get_logger(self, name, verbosity=2):
    msg_verbosity = ('verbosity option {} is invalid. Valid options are {}.'
        .format(verbosity, self.log_levels.keys()))
    assert verbosity in self.log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(self.log_levels[verbosity])
    return logger
