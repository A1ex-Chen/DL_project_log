def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': 
        False, 'formatters': {name: {'format': '%(message)s'}}, 'handlers':
        {name: {'class': 'logging.StreamHandler', 'formatter': name,
        'level': level}}, 'loggers': {name: {'level': level, 'handlers': [
        name], 'propagate': False}}})
