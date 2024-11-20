def load_dataset(self):
    self.preprocess()
    dataset_path = self._get_preprocessed_dataset_path()
    dataset = pickle.load(dataset_path.open('rb'))
    return dataset
