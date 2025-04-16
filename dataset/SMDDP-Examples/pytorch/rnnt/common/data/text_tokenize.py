def tokenize(self, transcript):
    if self.use_sentpiece:
        inds = self.sentpiece.encode(transcript, out_type=int)
        assert 0 not in inds, '<unk> found during tokenization (OOV?)'
    else:
        inds = [self.label2ind[x] for x in transcript if x in self.label2ind]
    return inds
