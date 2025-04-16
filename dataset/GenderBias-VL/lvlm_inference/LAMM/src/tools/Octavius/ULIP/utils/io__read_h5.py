@classmethod
def _read_h5(cls, file_path):
    f = h5py.File(file_path, 'r')
    return f['data'][()]
