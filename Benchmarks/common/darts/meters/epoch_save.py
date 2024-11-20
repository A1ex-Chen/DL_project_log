def save(self, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'{self.name}_epoch_results')
    self.dataframe().to_csv(path, index=False)
