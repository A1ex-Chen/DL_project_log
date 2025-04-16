def get_output_embeddings(self):
    if self.config.tie_word_embeddings:
        return self.shared
    else:
        return self.lm_head
