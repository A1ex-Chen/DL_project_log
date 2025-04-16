def load(self):
    logger.warning(f'Loading TensorRT engine: {self.engine_path}')
    self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
