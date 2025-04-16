def __init__(self, process_id: int, child_connection: Connection, interval:
    float):
    super().__init__()
    self.process_id = process_id
    self.interval = interval
    self.connection = child_connection
    self.num_measurements = 1
    self.mem_usage = get_cpu_memory(self.process_id)
