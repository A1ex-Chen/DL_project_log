def __init__(self, early_stopping_patience: int=1, early_stopping_threshold:
    Optional[float]=0.0):
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_threshold = early_stopping_threshold
    self.early_stopping_patience_counter = 0
