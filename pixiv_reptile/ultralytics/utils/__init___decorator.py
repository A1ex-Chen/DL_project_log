def decorator(func):
    """Decorator to apply temporary rc parameters and backend to a function."""

    def wrapper(*args, **kwargs):
        """Sets rc parameters and backend, calls the original function, and restores the settings."""
        original_backend = plt.get_backend()
        if backend.lower() != original_backend.lower():
            plt.close('all')
            plt.switch_backend(backend)
        with plt.rc_context(rcparams):
            result = func(*args, **kwargs)
        if backend != original_backend:
            plt.close('all')
            plt.switch_backend(original_backend)
        return result
    return wrapper
