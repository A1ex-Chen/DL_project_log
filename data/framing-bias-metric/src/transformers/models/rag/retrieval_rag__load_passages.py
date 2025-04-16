def _load_passages(self):
    logger.info('Loading passages from {}'.format(self.index_path))
    passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)
    with open(passages_path, 'rb') as passages_file:
        passages = pickle.load(passages_file)
    return passages
