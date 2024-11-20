def gen_c_embeddings(self, clip):
    clip = self.clip_mapper(clip)
    clip = self.seq_norm(clip)
    return clip
