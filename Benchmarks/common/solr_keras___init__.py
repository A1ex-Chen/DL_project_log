def __init__(self, timeout_in_sec=10):
    """Initialize TerminateOnTimeOut class.

        Parameters
        -----------
        timeout_in_sec : int
            seconds to timeout
        """
    super(TerminateOnTimeOut, self).__init__()
    self.run_timestamp = None
    self.timeout_in_sec = timeout_in_sec
