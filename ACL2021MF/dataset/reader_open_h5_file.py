def open_h5_file(self):
    self.features_h5 = h5py.File(self.features_h5path, 'r')
