def setup_logging(log_file=os.devnull):
    """
    Configures logging.
    By default logs from all workers are printed to the console, entries are
    prefixed with "N: " where N is the rank of the worker. Logs printed to the
    console don't include timestaps.
    Full logs with timestamps are saved to the log_file file.
    """


    class RankFilter(logging.Filter):

        def __init__(self, rank):
            self.rank = rank

        def filter(self, record):
            record.rank = self.rank
            return True
    rank = get_rank()
    rank_filter = RankFilter(rank)
    logging_format = '%(asctime)s - %(levelname)s - %(rank)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logging_format, datefmt
        ='%Y-%m-%d %H:%M:%S', filename=log_file, filemode='w')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(rank)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)
