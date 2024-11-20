def init_logger(self, log_dir=None, level=logging.INFO):
    """
        Init the logger.
        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.
        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
        level=level)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if log_dir and self.rank == 0:
        filename = '{}.log'.format(self.timestamp)
        log_file = os.path.join(log_dir, filename)
        self._add_file_handler(logger, log_file, level=level)
    return logger
