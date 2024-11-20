def set_logging(rank=-1):
    logging.basicConfig(format='%(message)s', level=logging.INFO if rank in
        [-1, 0] else logging.WARN)
