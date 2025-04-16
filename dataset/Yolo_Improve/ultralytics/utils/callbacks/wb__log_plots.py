def _log_plots(plots, step):
    """Logs plots from the input dictionary if they haven't been logged already at the specified step."""
    for name, params in plots.copy().items():
        timestamp = params['timestamp']
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp
