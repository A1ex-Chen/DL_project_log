def load_cached(self, path):
    """Load the data from disk"""
    frame = pd.read_csv(path)
    self.data = frame.pop('data')
    if len(frame.columns) > 1:
        self.labels = frame.to_dict()
    else:
        self.labels = frame['labels']
