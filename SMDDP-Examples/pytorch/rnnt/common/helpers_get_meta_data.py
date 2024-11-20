def get_meta_data(self, max_f_len, audio_shape_, transcripts,
    transcripts_lengths, async_cp=False, idx=0):
    self.meta_data = []
    B_split = transcripts.size(0) // self.batch_split_factor
    for i in range(self.batch_split_factor):
        self.meta_data.append(self.get_packing_meta_data(max_f_len,
            audio_shape_[i * B_split:(i + 1) * B_split],
            transcripts_lengths[i * B_split:(i + 1) * B_split], async_cp, i))
