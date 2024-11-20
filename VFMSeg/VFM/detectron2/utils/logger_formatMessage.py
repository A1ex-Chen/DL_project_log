def formatMessage(self, record):
    record.name = record.name.replace(self._root_name, self._abbrev_name)
    log = super(_ColorfulFormatter, self).formatMessage(record)
    if record.levelno == logging.WARNING:
        prefix = colored('WARNING', 'red', attrs=['blink'])
    elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
        prefix = colored('ERROR', 'red', attrs=['blink', 'underline'])
    else:
        return log
    return prefix + ' ' + log
