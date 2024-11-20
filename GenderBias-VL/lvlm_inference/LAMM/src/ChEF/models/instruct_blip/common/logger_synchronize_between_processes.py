def synchronize_between_processes(self):
    for meter in self.meters.values():
        meter.synchronize_between_processes()
