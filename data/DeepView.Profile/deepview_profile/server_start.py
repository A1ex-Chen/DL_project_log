def start(self):
    self._analysis_request_manager.start()
    self._connection_acceptor.start()
    logger.debug('DeepView server has started.')
