def wrapped_func(*args, **kwargs):
    """Applies retries to the decorated function or method."""
    self._attempts = 0
    while self._attempts < self.times:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._attempts += 1
            print(f'Retry {self._attempts}/{self.times} failed: {e}')
            if self._attempts >= self.times:
                raise e
            time.sleep(self.delay * 2 ** self._attempts)
