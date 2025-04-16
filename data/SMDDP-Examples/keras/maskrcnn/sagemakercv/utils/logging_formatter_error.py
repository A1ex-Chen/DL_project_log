def error(self, msg, *args, **kwargs):
    """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
    if self._logger is not None and self._logger.isEnabledFor(_Logger.ERROR):
        self._logger._log(_Logger.ERROR, msg, args, **kwargs)
