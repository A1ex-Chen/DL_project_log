def on_message_stderr(self, message):
    message = message.decode('ascii')
    if self.stderr_fd:
        self.stderr_fd.write(message)
    message = message.rstrip()
    print('stderr', message)
    if 'DeepView interactive profiling session started!' in message:
        self.state = 1
