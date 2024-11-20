def run_measure(self):
    for m in self.measurers:
        self.measurers[m].measurer_init()
    while self.measuring:
        for m in self.measurers:
            self.measurers[m].measurer_measure()
        time.sleep(self.sleep_interval)
    for m in self.measurers:
        self.measurers[m].measurer_deallocate()
