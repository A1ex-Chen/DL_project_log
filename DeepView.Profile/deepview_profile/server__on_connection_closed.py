def _on_connection_closed(self, address):
    self._main_executor.submit(self._connection_manager.remove_connection,
        address)
