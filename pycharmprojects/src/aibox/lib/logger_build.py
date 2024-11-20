@staticmethod
def build(name: str, path_to_log_file: str) ->'Logger':
    file_handler = logging.FileHandler(path_to_log_file)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.level = logging.INFO
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return Logger(logger, enabled=True)
