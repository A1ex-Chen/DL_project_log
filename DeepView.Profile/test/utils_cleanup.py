def cleanup(self):
    self.socket.close()
    self.listener_thread.join()
