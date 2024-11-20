@classmethod
def get(cls, file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension in ['.npy']:
        return cls._read_npy(file_path)
    elif file_extension in ['.pcd']:
        return cls._read_pcd(file_path)
    elif file_extension in ['.h5']:
        return cls._read_h5(file_path)
    elif file_extension in ['.txt']:
        return cls._read_txt(file_path)
    else:
        raise Exception('Unsupported file extension: %s' % file_extension)
