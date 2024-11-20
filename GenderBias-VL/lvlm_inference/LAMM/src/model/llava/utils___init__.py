def __init__(self, logger, log_level=logging.INFO):
    self.terminal = sys.stdout
    self.logger = logger
    self.log_level = log_level
    self.linebuf = ''
