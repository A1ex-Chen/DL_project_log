def _prepare_seq_split(self, audio, audio_shape, transcripts,
    transcripts_lengths):
    idx_sorted = torch.argsort(audio_shape, descending=True)
    audio_shape_sorted = audio_shape[idx_sorted]
    audio_sorted = audio[:, idx_sorted]
    transcripts_sorted = transcripts[idx_sorted]
    transcripts_lengths_sorted = transcripts_lengths[idx_sorted]
    batch_size = audio_shape.size(0)
    self.split_batch_size = batch_size // 2
    stack_factor = self.preproc.enc_stack_time_factor
    pivot_len = (audio_shape_sorted[self.split_batch_size] + stack_factor - 1
        ) // stack_factor * stack_factor
    self.pivot_len_cpu.copy_(pivot_len.detach(), non_blocking=True)
    return (audio_sorted, audio_shape_sorted, transcripts_sorted,
        transcripts_lengths_sorted)
