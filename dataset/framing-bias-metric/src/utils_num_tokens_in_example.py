def num_tokens_in_example(i):
    return min(self.src_lens[i], self.max_target_length)
