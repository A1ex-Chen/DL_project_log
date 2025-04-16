def cache_mem(self, curr_out, prev_mem):
    """cache hidden states into memory."""
    if self.reuse_len is not None and self.reuse_len > 0:
        curr_out = curr_out[:self.reuse_len]
    if prev_mem is None:
        new_mem = curr_out[-self.mem_len:]
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-self.mem_len:]
    return tf.stop_gradient(new_mem)
