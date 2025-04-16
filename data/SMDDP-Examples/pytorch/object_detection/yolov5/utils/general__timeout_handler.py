def _timeout_handler(self, signum, frame):
    raise TimeoutError(self.timeout_message)
