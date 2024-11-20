def CreateLogger(logger_name=LOGGER_NAME, log_level=LOG_LEVEL):
    if log_level == 'INFO':
        _log_level = logging.INFO
    elif log_level == 'WARN':
        _log_level = logging.WARN
    elif log_level == 'ERROR':
        _log_level = logging.ERROR
    else:
        _log_level = logging.DEBUG
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) > 0:
        logger.info('use existing logger')
        return logger
    logger.setLevel(_log_level)
    logger.propagate = True
    formatter = logging.Formatter(
        '[%(asctime)s:%(module)s:%(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    fileHandler = logging.handlers.RotatingFileHandler(filename=
        './debug.log', maxBytes=1048576, backupCount=5, mode='a')
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    fileHandler = logging.handlers.RotatingFileHandler(filename=
        './error.log', maxBytes=1048576, backupCount=5, mode='a')
    fileHandler.setLevel(logging.ERROR)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger
