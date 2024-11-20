def fetch_next(self):
    data = self.dali_it.__next__()
    audio, audio_shape = data[0]['audio'], data[0]['audio_shape'][:, 1]
    if audio.shape[0] == 0:
        return self.dali_it.__next__()
    if self.pipeline_type == 'val':
        audio = audio[:, :, :audio_shape.max()]
    transcripts, transcripts_lengths = self._gen_transcripts(data[0]['label'])
    if self.synthetic_text_seq_len != None:
        transcripts = torch.randint(transcripts.max(), (transcripts.size(0),
            self.synthetic_text_seq_len), device=transcripts.device, dtype=
            transcripts.dtype)
        transcripts_lengths = torch.ones_like(transcripts_lengths
            ) * self.synthetic_text_seq_len
    if self.enable_prefetch and self.preproc is not None:
        audio, audio_shape = self.preproc.preproc_func(audio, audio_shape)
        max_f_len = audio.size(0)
        if (self.pipeline_type == 'train' and self.min_seq_split_len > 0 and
            self.enable_prefetch):
            audio, audio_shape, transcripts, transcripts_lengths = (self.
                _prepare_seq_split(audio, audio_shape, transcripts,
                transcripts_lengths))
        self.preproc.get_meta_data(max_f_len, audio_shape, transcripts,
            transcripts_lengths, async_cp=True)
    return audio, audio_shape, transcripts, transcripts_lengths
