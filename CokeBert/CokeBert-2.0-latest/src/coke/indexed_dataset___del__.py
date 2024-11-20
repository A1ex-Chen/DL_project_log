def __del__(self):
    if self.data_file:
        self.data_file.close()
