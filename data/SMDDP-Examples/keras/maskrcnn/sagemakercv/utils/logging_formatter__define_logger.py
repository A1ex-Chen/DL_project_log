def _define_logger(self):
    if self._logger is not None:
        return self._logger
    with self._logger_lock:
        try:
            self._logger = _logging.getLogger(MODEL_NAME)
            self.reset_stream_handler()
        finally:
            self.set_verbosity(verbosity_level=_Logger.INFO)
        self._logger.propagate = False
