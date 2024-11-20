def _prepare_batch_for_alignment(self, sample, hypothesis):
    src_tokens = sample['net_input']['src_tokens']
    bsz = src_tokens.shape[0]
    src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1
        ).contiguous().view(bsz * self.beam_size, -1)
    src_lengths = sample['net_input']['src_lengths']
    src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous(
        ).view(bsz * self.beam_size)
    prev_output_tokens = data_utils.collate_tokens([beam['tokens'] for
        example in hypothesis for beam in example], self.pad, self.eos,
        self.left_pad_target, move_eos_to_beginning=True)
    tgt_tokens = data_utils.collate_tokens([beam['tokens'] for example in
        hypothesis for beam in example], self.pad, self.eos, self.
        left_pad_target, move_eos_to_beginning=False)
    return src_tokens, src_lengths, prev_output_tokens, tgt_tokens
