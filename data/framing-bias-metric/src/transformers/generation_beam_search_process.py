def process(self, input_ids: torch.LongTensor, next_scores: torch.
    FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.
    LongTensor, pad_token_id: Optional[int]=None, eos_token_id: Optional[
    int]=None) ->Tuple[torch.Tensor]:
    cur_len = input_ids.shape[-1]
    batch_size = len(self._beam_hyps)
    assert batch_size == input_ids.shape[0] // self.num_beams
    device = input_ids.device
    next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=
        next_scores.dtype, device=device)
    next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=
        next_tokens.dtype, device=device)
    next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=
        next_indices.dtype, device=device)
    for batch_idx, beam_hyp in enumerate(self._beam_hyps):
        if self._done[batch_idx]:
            assert len(beam_hyp
                ) >= self.num_beams, 'Batch can only be done if at least {} beams have been generated'.format(
                self.num_beams)
            assert eos_token_id is not None and pad_token_id is not None, 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
            next_beam_scores[batch_idx, :] = 0
            next_beam_tokens[batch_idx, :] = pad_token_id
            next_beam_indices[batch_idx, :] = 0
            continue
        beam_idx = 0
        for beam_token_rank, (next_token, next_score, next_index) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx],
            next_indices[batch_idx])):
            batch_beam_idx = batch_idx * self.num_beams + next_index
            if eos_token_id is not None and next_token.item() == eos_token_id:
                is_beam_token_worse_than_top_num_beams = (beam_token_rank >=
                    self.num_beams)
                if is_beam_token_worse_than_top_num_beams:
                    continue
                beam_hyp.add(input_ids[batch_beam_idx].clone(), next_score.
                    item())
            else:
                next_beam_scores[batch_idx, beam_idx] = next_score
                next_beam_tokens[batch_idx, beam_idx] = next_token
                next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                beam_idx += 1
            if beam_idx == self.num_beams:
                break
        if beam_idx < self.num_beams:
            raise ValueError(
                f'At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected.'
                )
        self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
            next_scores[batch_idx].max().item(), cur_len)
    return UserDict({'next_beam_scores': next_beam_scores.view(-1),
        'next_beam_tokens': next_beam_tokens.view(-1), 'next_beam_indices':
        next_beam_indices.view(-1)})
