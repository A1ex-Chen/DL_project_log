def __init__(self, labels=None, examples=None, mode='classification',
    verbose=False):
    self.labels = [] if labels is None else labels
    self.examples = [] if examples is None else examples
    self.mode = mode
    self.verbose = verbose
