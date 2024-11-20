def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
    self.check_avail()
    return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
