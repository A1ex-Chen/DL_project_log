def enable_progress_bar() ->None:
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True
