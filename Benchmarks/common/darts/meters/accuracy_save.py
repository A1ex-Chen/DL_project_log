def save(self, path, filename):
    """Save the task accuracies as a csv"""
    path = os.path.join(path, f'{filename}_accuracy.csv')
    self.dataframe().to_csv(path, index=False)
