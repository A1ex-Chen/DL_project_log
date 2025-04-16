def __init__(self, capture_io=True):
    self._logger = None
    self._logger_lock = threading.Lock()
    self._handlers = dict()
    self.old_warnings_showwarning = None
    if MPI_rank_and_size()[0] == 0:
        self._define_logger()
