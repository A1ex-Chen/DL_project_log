def get_time_values(self):
    """ Returns the time values.
        """
    if self.seq_len > 1:
        time = np.array([(i / (self.seq_len - 1)) for i in range(self.
            seq_len)], dtype=np.float32)
    else:
        time = np.array([1]).astype(np.float32)
    return time
