def setup_lmdb(self) ->bool:
    path_to_lmdb_dir = os.path.join(self.path_to_data_dir, 'lmdb')
    if os.path.exists(path_to_lmdb_dir):
        self._lmdb_env = lmdb.open(path_to_lmdb_dir)
        self._lmdb_txn = self._lmdb_env.begin()
        return True
    else:
        return False
