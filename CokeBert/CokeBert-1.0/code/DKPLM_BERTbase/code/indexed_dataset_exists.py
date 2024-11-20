@staticmethod
def exists(path):
    return os.path.exists(index_file_path(path)) and os.path.exists(
        data_file_path(path))
