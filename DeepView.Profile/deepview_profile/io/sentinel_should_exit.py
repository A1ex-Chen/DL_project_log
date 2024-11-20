def should_exit(self, ready_descriptors):
    return self._read_pipe in ready_descriptors
