def info(self, msg, *args, **kwargs):
    """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
    if self._logger is not None and self._logger.isEnabledFor(_Logger.INFO):
        self._logger._log(_Logger.INFO, msg, args, **kwargs)
