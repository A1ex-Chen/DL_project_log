def monitor_system_metrics(self):
    while self.running:
        current_metrics = self.get_system_metrics()
        for i, j in current_metrics.items():
            self.system_metrics[i].append(j)
        sleep(self.collection_interval)
