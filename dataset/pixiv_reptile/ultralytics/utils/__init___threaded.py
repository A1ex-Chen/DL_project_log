def threaded(func):
    """
    Multi-threads a target function by default and returns the thread or function result.

    Use as @threaded decorator. The function runs in a separate thread unless 'threaded=False' is passed.
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function based on 'threaded' kwarg and returns the thread or function result."""
        if kwargs.pop('threaded', True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs,
                daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)
    return wrapper
