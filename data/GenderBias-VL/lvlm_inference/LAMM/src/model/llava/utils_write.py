def write(self, buf):
    temp_linebuf = self.linebuf + buf
    self.linebuf = ''
    for line in temp_linebuf.splitlines(True):
        if line[-1] == '\n':
            self.logger.log(self.log_level, line.rstrip())
        else:
            self.linebuf += line
