def shutdown():
    self._connection_acceptor.stop()
    self._connection_manager.stop()
