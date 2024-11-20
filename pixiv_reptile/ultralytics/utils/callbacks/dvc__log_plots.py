def _log_plots(plots, prefix=''):
    """Logs plot images for training progress if they have not been previously processed."""
    for name, params in plots.items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            _log_images(name, prefix)
            _processed_plots[name] = timestamp
