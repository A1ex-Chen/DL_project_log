def collate_fn(batch):
    bs = len(batch)
    max_len = lambda l, idx: max(el[idx].size(0) for el in l)
    audio = torch.zeros(bs, max_len(batch, 0))
    audio_lens = torch.zeros(bs, dtype=torch.int32)
    transcript = torch.zeros(bs, max_len(batch, 2))
    transcript_lens = torch.zeros(bs, dtype=torch.int32)
    for i, sample in enumerate(batch):
        audio[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
        audio_lens[i] = sample[1]
        transcript[i].narrow(0, 0, sample[2].size(0)).copy_(sample[2])
        transcript_lens[i] = sample[3]
    return audio, audio_lens, transcript, transcript_lens
