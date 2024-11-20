def reset_stream_handler(self):
    if self._logger is None:
        raise RuntimeError(
            'Impossible to set handlers if the Logger is not predefined')
    try:
        self._logger.removeHandler(self._handlers['stream_stdout'])
    except KeyError:
        pass
    try:
        self._logger.removeHandler(self._handlers['stream_stderr'])
    except KeyError:
        pass
    self._handlers['stream_stdout'] = _logging.StreamHandler(sys.stdout)
    self._handlers['stream_stdout'].addFilter(lambda record: record.levelno <=
        _logging.INFO)
    self._handlers['stream_stderr'] = _logging.StreamHandler(sys.stderr)
    self._handlers['stream_stderr'].addFilter(lambda record: record.levelno >
        _logging.INFO)
    Formatter = StdOutFormatter
    self._handlers['stream_stdout'].setFormatter(Formatter())
    self._logger.addHandler(self._handlers['stream_stdout'])
    try:
        self._handlers['stream_stderr'].setFormatter(Formatter())
        self._logger.addHandler(self._handlers['stream_stderr'])
    except KeyError:
        pass
