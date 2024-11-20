def cache_mem(self, curr_out, prev_mem):
    if self.reuse_len is not None and self.reuse_len > 0:
        curr_out = curr_out[:self.reuse_len]
    if self.mem_len is None or self.mem_len == 0:
        cutoff = 0
    else:
        cutoff = -self.mem_len
    if prev_mem is None:
        new_mem = curr_out[cutoff:]
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[cutoff:]
    return tf.stop_gradient(new_mem)
