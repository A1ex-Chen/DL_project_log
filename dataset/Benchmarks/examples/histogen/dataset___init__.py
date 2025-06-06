def __init__(self, path):
    self.env = lmdb.open(path, max_readers=32, readonly=True, lock=False,
        readahead=False, meminit=False)
    if not self.env:
        raise IOError('Cannot open lmdb dataset', path)
    with self.env.begin(write=False) as txn:
        self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
