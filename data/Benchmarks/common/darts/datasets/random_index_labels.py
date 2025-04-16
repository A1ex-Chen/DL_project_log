def index_labels(self, idx):
    """Index into the labels"""
    return {key: value[idx] for key, value in self.labels.items()}
