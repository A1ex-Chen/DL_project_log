def add_weight_size(self, size_bytes, grad_size_bytes):
    self._size_bytes += size_bytes
    self._grad_size_bytes += grad_size_bytes
