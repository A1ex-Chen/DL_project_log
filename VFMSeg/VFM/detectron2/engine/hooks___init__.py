def __init__(self, period=20, max_runs=10):
    """
        Args:
            period (int): Output stats each 'period' iterations
            max_runs (int): Stop the logging after 'max_runs'
        """
    self._logger = logging.getLogger(__name__)
    self._period = period
    self._max_runs = max_runs
    self._runs = 0
