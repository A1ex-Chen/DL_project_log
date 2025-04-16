def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
    self._chunk_size = chunk_size
    self._chunk_dim = 1
