def __post_init__(self):
    if self.log_history is None:
        self.log_history = []
