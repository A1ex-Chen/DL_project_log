def on_message_stdout(self, message):
    message = message.decode('ascii')
    if self.stdout_fd:
        self.stdout_fd.write(message)
