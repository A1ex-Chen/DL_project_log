def __init__(self, target, width=30, verbose=1, interval=0.01):
    """
        Parameters
        ------------
        target: int
            total number of steps expected
        interval: float
            minimum visual progress update interval (in seconds)
        """
    self.width = width
    self.target = target
    self.sum_values = {}
    self.unique_values = []
    self.start = time.time()
    self.last_update = 0
    self.interval = interval
    self.total_width = 0
    self.seen_so_far = 0
    self.verbose = verbose
