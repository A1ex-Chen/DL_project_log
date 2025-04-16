def terminate(self):
    self.process.terminate()
    self.stdout_thread.join()
    self.stderr_thread.join()
