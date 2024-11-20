def create_accumulator(self, input_path, information_file=None):
    mm.metrics
    file_name = os.path.split(input_path)[1]
    self.frame_number = 1
    self.paths = np.hstack((self.paths, input_path))
    self.matrix_predictions = []
    if information_file is None:
        seqinfo_path = os.path.join(input_path, 'seqinfo.ini')
        information_file = InformationFile(file_path=seqinfo_path)
    length = information_file.search(variable_name='seqLength')
    self.progress_bar_iter = track(range(length - 1), description=file_name,
        transient=False)
