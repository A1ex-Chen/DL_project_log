def critical(self, msg, *args, **kwargs):
    """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
    if self._logger is not None and self._logger.isEnabledFor(_Logger.CRITICAL
        ):
        self._logger._log(_Logger.CRITICAL, msg, args, **kwargs)
