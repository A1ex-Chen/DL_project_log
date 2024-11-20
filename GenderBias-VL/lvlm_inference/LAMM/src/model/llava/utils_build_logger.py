def build_logger(logger_name, logger_filename):
    global handler
    formatter = logging.Formatter(fmt=
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt=
        '%Y-%m-%d %H:%M:%S')
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)
    stdout_logger = logging.getLogger('stdout')
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger('stderr')
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when=
            'D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)
        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)
    return logger
