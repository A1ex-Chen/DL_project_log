def flush(self):
    if self.linebuf != '':
        self.logger.log(self.log_level, self.linebuf.rstrip())
    self.linebuf = ''
