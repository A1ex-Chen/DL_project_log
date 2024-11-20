def stop(self):

    def shutdown():
        self._connection_acceptor.stop()
        self._connection_manager.stop()
    self._analysis_request_manager.stop()
    self._main_executor.submit(shutdown).result()
    self._main_executor.shutdown()
    logger.debug('DeepView server has shut down.')
