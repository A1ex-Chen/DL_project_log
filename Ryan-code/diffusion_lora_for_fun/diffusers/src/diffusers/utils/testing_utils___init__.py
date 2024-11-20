def __init__(self, logger):
    self.logger = logger
    self.io = StringIO()
    self.sh = logging.StreamHandler(self.io)
    self.out = ''
