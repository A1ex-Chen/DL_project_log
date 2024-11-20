@functools.lru_cache()
def setup_logger(output=None, distributed_rank=0, *, color=True, name=
    'detectron2', abbrev_name=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if abbrev_name is None:
        abbrev_name = 'd2' if name == 'detectron2' else name
    plain_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt=
        '%m/%d %H:%M:%S')
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(colored(
                '[%(asctime)s %(name)s]: ', 'green') + '%(message)s',
                datefmt='%m/%d %H:%M:%S', root_name=name, abbrev_name=str(
                abbrev_name))
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            filename = output
        else:
            filename = os.path.join(output, 'log.txt')
        if distributed_rank > 0:
            filename = filename + '.rank{}'.format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    return logger
