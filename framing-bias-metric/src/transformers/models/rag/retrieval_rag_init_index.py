def init_index(self):
    if not self.is_initialized():
        logger.info('Loading index from {}'.format(self.index_path))
        self.dataset.load_faiss_index('embeddings', file=self.index_path)
        self._index_initialized = True
