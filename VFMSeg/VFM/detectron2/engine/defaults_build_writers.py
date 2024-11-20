def build_writers(self):
    """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
    return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
