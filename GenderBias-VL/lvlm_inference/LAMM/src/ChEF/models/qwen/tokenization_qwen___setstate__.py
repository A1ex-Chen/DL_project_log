def __setstate__(self, state):
    self.__dict__.update(state)
    enc = tiktoken.Encoding('Qwen', pat_str=PAT_STR, mergeable_ranks=self.
        mergeable_ranks, special_tokens=self.special_tokens)
    self.tokenizer = enc
