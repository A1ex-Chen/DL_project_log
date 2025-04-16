@property
def listening_on(self):
    return self._connection_acceptor.host, self._connection_acceptor.port
