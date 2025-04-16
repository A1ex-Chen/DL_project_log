def __init__(self, patience=5, verbose=False, delta=0, path=
    '../chap08/data/checkpoint.pt'):
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.delta = delta
    self.path = path
