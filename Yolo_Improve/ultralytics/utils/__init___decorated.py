@wraps(f)
def decorated(*args, **kwargs):
    """Applies thread-safety to the decorated function or method."""
    with self.lock:
        return f(*args, **kwargs)
