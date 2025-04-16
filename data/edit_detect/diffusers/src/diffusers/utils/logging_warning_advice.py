def warning_advice(self, *args, **kwargs) ->None:
    """
    This method is identical to `logger.warning()`, but if env var DIFFUSERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv('DIFFUSERS_NO_ADVISORY_WARNINGS', False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)
