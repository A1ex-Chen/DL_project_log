def start_monitoring(self):
    self.running = True
    self.thread = threading.Thread(target=self.monitor_system_metrics)
    self.thread.start()
