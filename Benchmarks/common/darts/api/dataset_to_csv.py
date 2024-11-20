def to_csv(self, path):
    """Save the data to disk"""
    self.dataframe().to_csv(path, index=False)
