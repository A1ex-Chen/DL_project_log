def finalize(self, input_ids: torch.LongTensor, final_beam_scores: torch.
    FloatTensor, final_beam_tokens: torch.LongTensor, final_beam_indices:
    torch.LongTensor, pad_token_id: Optional[int]=None, eos_token_id:
    Optional[int]=None) ->torch.LongTensor:
    batch_size = len(self._state_beam_hyps)
    for batch_idx, state_beam_hyp in enumerate(self._state_beam_hyps):
        for state_idx, beam_hyp in enumerate(state_beam_hyp):
            if self._done[batch_idx, state_idx]:
                continue
            start_index = (batch_idx * self.num_beams * self.num_states + 
                state_idx * self.num_beams)
            for beam_id in range(self.num_beams):
                batch_beam_idx = start_index + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)
    sent_lengths = input_ids.new(batch_size * self.num_states * self.
        num_beam_hyps_to_keep)
    best = []
    for i, state_beam_hyp in enumerate(self._state_beam_hyps):
        for ii, beam_hyp in enumerate(state_beam_hyp):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[self.num_beam_hyps_to_keep * self.num_states *
                    i + ii * self.num_beam_hyps_to_keep + j] = len(best_hyp)
                best.append(best_hyp)
    sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
    decoded: torch.LongTensor = input_ids.new(batch_size * self.num_states *
        self.num_beam_hyps_to_keep, sent_max_len)
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, '`pad_token_id` has to be defined'
        decoded.fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, :sent_lengths[i]] = hypo
        if sent_lengths[i] < self.max_length:
            decoded[i, sent_lengths[i]] = eos_token_id
    return decoded
