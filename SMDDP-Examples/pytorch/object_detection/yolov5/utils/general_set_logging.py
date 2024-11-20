def set_logging(name=None, verbose=VERBOSE):
    if is_kaggle():
        for h in logging.root.handlers:
            logging.root.removeHandler(h)
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    handler.setLevel(level)
    log.addHandler(handler)
