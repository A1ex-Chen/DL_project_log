def load_files(self, model_path, start_idx):
    """ Loads the model files.

        Args:
            model_path (str): path to model
            start_idx (int): id of sequence start
        """
    folder = os.path.join(model_path, self.folder_name)
    files = glob.glob(os.path.join(folder, '*.npz'))
    files.sort()
    files = files[start_idx:start_idx + self.seq_len]
    if self.only_end_points:
        files = [files[0], files[-1]]
    return files
