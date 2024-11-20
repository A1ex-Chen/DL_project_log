def __getitem__(self, index):
    with self.env.begin(write=False) as txn:
        key = str(index).encode('utf-8')
        row = pickle.loads(txn.get(key))
    return torch.from_numpy(row.top), torch.from_numpy(row.bottom
        ), row.filename
