def process(self, input_ids: torch.LongTensor, next_scores: torch.
    FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.
    LongTensor, pad_token_id: Optional[int]=None, eos_token_id: Optional[
    int]=None) ->Tuple[torch.Tensor]:
    cur_len = input_ids.shape[-1]
    batch_size = len(self._state_beam_hyps)
    device = input_ids.device
    next_beam_scores = torch.ones((batch_size, self.num_states, self.
        num_beams), dtype=next_scores.dtype, device=device) * -1000000000.0
    next_beam_tokens = torch.zeros((batch_size, self.num_states, self.
        num_beams), dtype=next_tokens.dtype, device=device)
    next_beam_indices = torch.zeros((batch_size, self.num_states, self.
        num_beams), dtype=next_indices.dtype, device=device)
    for batch_idx, state_beam_hyp in enumerate(self._state_beam_hyps):
        for state_idx, beam_hyp in enumerate(state_beam_hyp):
            if self._done[batch_idx, state_idx]:
                assert len(beam_hyp
                    ) >= self.num_beams, 'Batch can only be done if at least {} beams have been generated'.format(
                    self.num_beams)
                assert eos_token_id is not None and pad_token_id is not None, 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                next_beam_scores[batch_idx, state_idx, :] = -1000000000.0
                next_beam_tokens[batch_idx, state_idx, :] = pad_token_id
                next_beam_indices[batch_idx, state_idx, :] = 0
                continue
            beam_index = batch_idx * self.num_states + state_idx
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index
                ) in enumerate(zip(next_tokens[beam_index], next_scores[
                beam_index], next_indices[beam_index])):
                batch_beam_idx = (batch_idx * self.num_beams * self.
                    num_states + next_index)
                if eos_token_id is not None and next_token.item(
                    ) == eos_token_id and next_score.item() > -1000000000.0:
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= self.num_beams)
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx].clone(),
                        next_score.item())
                else:
                    next_beam_scores[batch_idx, state_idx, beam_idx
                        ] = next_score
                    next_beam_tokens[batch_idx, state_idx, beam_idx
                        ] = next_token
                    next_beam_indices[batch_idx, state_idx, beam_idx
                        ] = batch_beam_idx
                    beam_idx += 1
                if beam_idx == self.num_beams:
                    break
        for state_idx, beam_hyp in enumerate(state_beam_hyp):
            beam_index = batch_idx * self.num_states + state_idx
            self._done[batch_idx, state_idx] = self._done[batch_idx, state_idx
                ] or beam_hyp.is_done(next_scores[beam_index].max().item(),
                cur_len)
    return UserDict({'next_beam_scores': next_beam_scores.view(-1),
        'next_beam_tokens': next_beam_tokens.view(-1), 'next_beam_indices':
        next_beam_indices.view(-1)})
