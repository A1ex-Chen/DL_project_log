def teardown_lmdb(self):
    if self._lmdb_env is not None:
        self._lmdb_env.close()
