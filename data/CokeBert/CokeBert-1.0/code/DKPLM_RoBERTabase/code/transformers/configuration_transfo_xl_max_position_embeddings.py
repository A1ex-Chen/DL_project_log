@property
def max_position_embeddings(self):
    return self.tgt_len + self.ext_len + self.mem_len
