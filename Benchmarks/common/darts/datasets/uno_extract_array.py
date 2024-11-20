@staticmethod
def extract_array(path, remove_finished=False):
    print('Extracting {}'.format(path))
    arry = np.load(path)
    if remove_finished:
        os.unlink(path)
