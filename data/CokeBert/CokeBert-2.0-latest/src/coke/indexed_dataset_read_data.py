def read_data(self, path):
    self.data_file = open(data_file_path(path), 'rb', buffering=0)
