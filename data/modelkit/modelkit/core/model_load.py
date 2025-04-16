def load(self) ->None:
    """Load dependencies before loading the asset"""
    try:
        sniffio.current_async_library()
        async_context = True
    except sniffio.AsyncLibraryNotFoundError:
        async_context = False
    for model_name, m in self.model_dependencies.items():
        if not m._loaded:
            m.load()
        if not async_context and isinstance(m, AsyncModel):
            self.model_dependencies[model_name] = WrappedAsyncModel(m)
    with PerformanceTracker() as m:
        self._load()
    logger.debug('Model loaded', model_name=self.configuration_key, time=
        humanize.naturaldelta(m.time, minimum_unit='seconds'), time_s=m.
        time, memory=humanize.naturalsize(m.increment) if m.increment is not
        None else None, memory_bytes=m.increment)
    self._loaded = True
    self._load_time = m.time
    self._load_memory_increment = m.increment
