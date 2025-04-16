def __getitem__(self, idx):
    src = torch.tensor(self.tokenizer.segment(self.raw_src[idx]))
    tgt = torch.tensor(self.tokenizer.segment(self.raw_tgt[idx]))
    return src, tgt
