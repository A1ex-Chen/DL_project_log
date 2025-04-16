def __init__(self, host, port, handler_function):
    self._host = host
    self._port = port
    self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self._handler_function = handler_function
    self._acceptor = Thread(target=self._accept_connections)
    self._sentinel = Sentinel()
