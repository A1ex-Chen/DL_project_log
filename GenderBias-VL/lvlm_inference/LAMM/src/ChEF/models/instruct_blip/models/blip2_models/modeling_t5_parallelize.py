@add_start_docstrings(PARALLELIZE_DOCSTRING)
def parallelize(self, device_map=None):
    self.device_map = get_device_map(len(self.encoder.block), range(torch.
        cuda.device_count())) if device_map is None else device_map
    assert_device_map(self.device_map, len(self.encoder.block))
    self.encoder.parallelize(self.device_map)
    self.model_parallel = True
