def __init__(self, start_iter=0):
    """
        Args:
            start_iter (int): the iteration number to start with
        """
    self._history = defaultdict(HistoryBuffer)
    self._smoothing_hints = {}
    self._latest_scalars = {}
    self._iter = start_iter
    self._current_prefix = ''
    self._vis_data = []
    self._histograms = []
