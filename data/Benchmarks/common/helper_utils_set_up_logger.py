def set_up_logger(logfile, logger, verbose=False, fmt_line=
    '[%(asctime)s %(process)d] %(message)s', fmt_date='%Y-%m-%d %H:%M:%S'):
    """Set up the event logging system. Two handlers are created.
    One to send log records to a specified file and
    one to send log records to the (defaulf) sys.stderr stream.
    The logger and the file handler are set to DEBUG logging level.
    The stream handler is set to INFO logging level, or to DEBUG
    logging level if the verbose flag is specified.
    Logging messages which are less severe than the level set will
    be ignored.

    Parameters
    ----------
    logfile : filename
        File to store the log records
    logger : logger object
        Python object for the logging interface
    verbose : boolean
        Flag to increase the logging level from INFO to DEBUG. It
        only applies to the stream handler.
    """
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter(fmt_line, datefmt=fmt_date))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
