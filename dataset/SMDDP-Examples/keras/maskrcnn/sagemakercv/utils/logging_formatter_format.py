def format(self, record):
    try:
        message = record.getMessage()
        assert isinstance(message, str)
        record.message = self.to_unicode(message)
    except Exception as e:
        record.message = 'Bad message (%r): %r' % (e, record.__dict__)
    record.asctime = self.formatTime(record, self.datefmt)
    if record.levelno in self._colors:
        record.color = self._colors[record.levelno]
        record.end_color = self._normal
    else:
        record.color = record.end_color = ''
    formatted = self._fmt % record.__dict__
    if record.exc_info:
        if not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
    if record.exc_text:
        lines = [formatted.rstrip()]
        lines.extend(self.to_unicode(ln) for ln in record.exc_text.split('\n'))
        formatted = '\n'.join(lines)
    return formatted.replace('\n', '\n    ')
