def tokenize(self, transcripts):
    transcripts = [transcripts[i] for i in range(len(transcripts))]
    if self.normalize_transcripts:
        transcripts = [normalize_string(t, self.tokenizer.charset,
            punctuation_map(self.tokenizer.charset)) for t in transcripts]
    if not self.tokenized_transcript:
        transcripts = [self.tokenizer.tokenize(t) for t in transcripts]
    if self.jit_tensor_formation:
        self.tr = transcripts
    else:
        self.tr = np.empty(len(transcripts), dtype=object)
        for i in range(len(transcripts)):
            self.tr[i] = torch.tensor(transcripts[i])
    self.t_sizes = torch.tensor([len(t) for t in transcripts], dtype=torch.
        int32)
    self.max_txt_len = self.t_sizes.max().item()
