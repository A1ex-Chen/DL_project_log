def debug(self, msg, *args, **kwargs):
    """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
    if self._logger is not None and self._logger.isEnabledFor(_Logger.DEBUG):
        self._logger._log(_Logger.DEBUG, msg, args, **kwargs)
