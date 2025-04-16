def setup_logging(log_all_ranks=True, filename=os.devnull, filemode='w'):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """


    class RankFilter(logging.Filter):

        def __init__(self, rank, log_all_ranks):
            self.rank = rank
            self.log_all_ranks = log_all_ranks

        def filter(self, record):
            record.rank = self.rank
            if self.log_all_ranks:
                return True
            else:
                return self.rank == 0
    rank = utils.distributed.get_rank()
    rank_filter = RankFilter(rank, log_all_ranks)
    if log_all_ranks:
        logging_format = '%(asctime)s - %(levelname)s - %(rank)s - %(message)s'
    else:
        logging_format = '%(asctime)s - %(levelname)s - %(message)s'
        if rank != 0:
            filename = os.devnull
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(level=logging.DEBUG, format=logging_format, datefmt
        ='%Y-%m-%d %H:%M:%S', filename=filename, filemode=filemode)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    if log_all_ranks:
        formatter = logging.Formatter('%(rank)s: %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)
