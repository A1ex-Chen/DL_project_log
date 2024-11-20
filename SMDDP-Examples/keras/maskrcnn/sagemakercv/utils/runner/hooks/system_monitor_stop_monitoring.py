def stop_monitoring(self):
    self.running = False
    self.thread.join()
