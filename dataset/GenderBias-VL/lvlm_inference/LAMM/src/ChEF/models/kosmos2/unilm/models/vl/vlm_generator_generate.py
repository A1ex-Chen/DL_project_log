@torch.no_grad()
def generate(self, models, sample, **kwargs):
    finalized = super()._generate(sample, **kwargs)
    src_tokens = sample['net_input']['src_tokens']
    bsz = src_tokens.shape[0]
    beam_size = self.beam_size
    src_tokens, src_lengths, prev_output_tokens, tgt_tokens = (self.
        _prepare_batch_for_alignment(sample, finalized))
    if any(getattr(m, 'full_context_alignment', False) for m in self.model.
        models):
        attn = self.model.forward_align(src_tokens, src_lengths,
            prev_output_tokens)
    else:
        attn = [finalized[i // beam_size][i % beam_size]['attention'].
            transpose(1, 0) for i in range(bsz * beam_size)]
    if src_tokens.device != 'cpu':
        src_tokens = src_tokens.to('cpu')
        tgt_tokens = tgt_tokens.to('cpu')
        attn = [i.to('cpu') for i in attn]
    for i in range(bsz * beam_size):
        alignment = self.extract_alignment(attn[i], src_tokens[i],
            tgt_tokens[i], self.pad, self.eos)
        finalized[i // beam_size][i % beam_size]['alignment'] = alignment
    return finalized
