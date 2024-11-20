def _load(self, model_name):
    """
        This function loads a configured model by name.
        """
    with PerformanceTracker() as m:
        self._check_configurations(model_name)
        self._resolve_assets(model_name)
        self._load_model(model_name)
    logger.info('Model and dependencies loaded', name=model_name, time=
        humanize.naturaldelta(m.time, minimum_unit='seconds'), time_s=m.
        time, memory=humanize.naturalsize(m.increment) if m.increment is not
        None else None, memory_bytes=m.increment)
