def connect(self, addr, port):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.connect((addr, port))
    self.listener_thread = threading.Thread(target=socket_monitor, args=(
        self.socket, self.handle_message))
    self.listener_thread.start()
