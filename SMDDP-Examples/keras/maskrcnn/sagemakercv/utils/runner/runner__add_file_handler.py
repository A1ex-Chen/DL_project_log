def _add_file_handler(self, logger, filename=None, mode='w', level=logging.INFO
    ):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    file_handler.setStream(sys.stdout)
    logger.addHandler(file_handler)
    return logger
