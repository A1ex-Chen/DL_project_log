def run(self):
    self.connection.send(0)
    stop = False
    while True:
        self.mem_usage = max(self.mem_usage, get_cpu_memory(self.process_id))
        self.num_measurements += 1
        if stop:
            break
        stop = self.connection.poll(self.interval)
    self.connection.send(self.mem_usage)
    self.connection.send(self.num_measurements)
