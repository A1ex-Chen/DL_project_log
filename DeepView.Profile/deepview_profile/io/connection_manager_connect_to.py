def connect_to(self, host, port):
    new_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    new_socket.connect((host, port))
    self.register_connection(new_socket, (host, port))
