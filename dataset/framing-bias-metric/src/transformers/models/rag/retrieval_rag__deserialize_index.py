def _deserialize_index(self):
    logger.info('Loading index from {}'.format(self.index_path))
    resolved_index_path = self._resolve_path(self.index_path, self.
        INDEX_FILENAME + '.index.dpr')
    self.index = faiss.read_index(resolved_index_path)
    resolved_meta_path = self._resolve_path(self.index_path, self.
        INDEX_FILENAME + '.index_meta.dpr')
    with open(resolved_meta_path, 'rb') as metadata_file:
        self.index_id_to_db_id = pickle.load(metadata_file)
    assert len(self.index_id_to_db_id
        ) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'
