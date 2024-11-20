def tokenize(self, line, add_eos=False, add_double_eos=False):
    return self.tokenizer.encode(line)
