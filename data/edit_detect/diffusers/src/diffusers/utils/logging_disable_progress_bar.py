def disable_progress_bar() ->None:
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
