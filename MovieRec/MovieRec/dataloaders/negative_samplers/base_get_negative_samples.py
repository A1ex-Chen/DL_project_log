def get_negative_samples(self):
    savefile_path = self._get_save_path()
    if savefile_path.is_file():
        print('Negatives samples exist. Loading.')
        negative_samples = pickle.load(savefile_path.open('rb'))
        return negative_samples
    print("Negative samples don't exist. Generating.")
    negative_samples = self.generate_negative_samples()
    with savefile_path.open('wb') as f:
        pickle.dump(negative_samples, f)
    return negative_samples
