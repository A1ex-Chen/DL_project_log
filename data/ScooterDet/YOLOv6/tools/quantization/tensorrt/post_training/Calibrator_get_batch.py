def get_batch(self, names):
    try:
        batch = next(self.batches)
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]
    except StopIteration:
        return None
