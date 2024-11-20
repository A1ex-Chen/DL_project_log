def __next__(self):
    if self.enable_prefetch:
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        self.prefetch_stream.synchronize()
        if self.prefetched_data is None:
            raise StopIteration
        else:
            for i, packed_batch_cpu in enumerate(self.preproc.
                list_packed_batch_cpu):
                self.preproc.meta_data[i]['packed_batch'
                    ] = packed_batch_cpu.item()
            if self.pipeline_type == 'train' and self.min_seq_split_len > 0:
                audio, audio_shape, transcripts, transcripts_lengths = (self
                    .prefetched_data)
                second_segment_len = audio.size(0) - self.pivot_len_cpu
                if second_segment_len >= self.min_seq_split_len:
                    list_audio = [audio[:self.pivot_len_cpu], audio[self.
                        pivot_len_cpu:, :self.split_batch_size]]
                    return (list_audio, audio_shape, transcripts,
                        transcripts_lengths)
            return self.prefetched_data
    else:
        return self.fetch_next()
