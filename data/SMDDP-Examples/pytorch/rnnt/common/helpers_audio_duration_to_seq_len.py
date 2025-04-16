def audio_duration_to_seq_len(self, audio_duration, after_subsampling,
    after_stack_time):
    if after_stack_time:
        assert after_subsampling, 'after_stacktime == True while after_subsampling == False is not a valid use case'
    seq_len = audio_duration // self.window_stride + 1
    if after_subsampling == False:
        return seq_len
    else:
        seq_len_sub_sampled = math.ceil(seq_len / self.frame_subsampling)
        if after_stack_time == False:
            return seq_len_sub_sampled
        else:
            return (seq_len_sub_sampled + self.enc_stack_time_factor - 1
                ) // self.enc_stack_time_factor
