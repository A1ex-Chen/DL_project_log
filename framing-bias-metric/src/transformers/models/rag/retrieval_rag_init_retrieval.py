def init_retrieval(self):
    """
        Retriever initalization function. It loads the index into memory.
        """
    logger.info('initializing retrieval')
    self.index.init_index()
