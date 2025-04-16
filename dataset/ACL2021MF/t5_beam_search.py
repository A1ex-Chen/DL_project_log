def beam_search(self, input_ids: torch.LongTensor, beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList]=None, max_length:
    Optional[int]=None, state_transition: Optional[torch.Tensor]=None,
    pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=None, **
    model_kwargs):
    logits_processor = (logits_processor if logits_processor is not None else
        LogitsProcessorList())
    max_length = (max_length if max_length is not None else self.config.
        max_length)
    pad_token_id = (pad_token_id if pad_token_id is not None else self.
        config.pad_token_id)
    eos_token_id = (eos_token_id if eos_token_id is not None else self.
        config.eos_token_id)
    if state_transition is None:
        batch_size = len(beam_scorer._beam_hyps)
    else:
        batch_size = len(beam_scorer._state_beam_hyps)
    num_beams = beam_scorer.num_beams
    use_constrained_decoding = state_transition is not None
    num_states = 1
    if state_transition is not None:
        assert state_transition.size(1) == state_transition.size(2
            ), 'num state wrong'
        num_states = state_transition.size(1)
        state_transition = state_transition.unsqueeze(3).expand((-1, -1, -1,
            num_beams, -1))
    batch_beam_size, cur_len = input_ids.shape
    assert num_beams * num_states * batch_size == batch_beam_size, 'Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}.'
    if not use_constrained_decoding:
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.
            float, device=input_ids.device)
        beam_scores[:, 1:] = -1000000000.0
    else:
        beam_scores = torch.zeros((batch_size, num_states, num_beams),
            dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1000000000.0
        beam_scores[:, 0, 1:] = -1000000000.0
    beam_scores = beam_scores.view((-1,))
    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **
            model_kwargs)
        outputs = self(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len, max_length=max_length)
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = logits_processor(input_ids, next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores[outputs.repeat_mask] = -1000000000.0
        if use_constrained_decoding:
            next_token_scores = next_token_scores.view(batch_size,
                num_states, num_beams, -1)
            beam_scores = beam_scores.view(batch_size, num_states, num_beams)
            constrained_beam_scores = torch.FloatTensor(batch_size,
                num_states, 2 * num_beams).to(next_token_scores.device)
            constrained_beam_indices = torch.LongTensor(batch_size,
                num_states, 2 * num_beams).to(next_token_scores.device)
            for i in range(num_states):
                cloned_scores = next_token_scores.clone()
                cloned_scores[~state_transition[:, :, i, :, :]] = -1000000000.0
                overall_scores = cloned_scores + beam_scores[:, :, :, None
                    ].expand_as(cloned_scores)
                overall_scores = overall_scores.view(batch_size, -1)
                state_beam_log_probs, state_beam_indices = torch.topk(
                    overall_scores, 2 * num_beams, dim=1, largest=True,
                    sorted=True)
                constrained_beam_scores[:, i, :] = state_beam_log_probs
                constrained_beam_indices[:, i, :] = state_beam_indices
            next_token_scores, next_tokens = constrained_beam_scores.view(-
                1, 2 * num_beams), constrained_beam_indices.view(-1, 2 *
                num_beams)
        else:
            next_token_scores = next_token_scores + beam_scores[:, None
                ].expand_as(next_token_scores)
            next_token_scores = next_token_scores.view(batch_size, 
                num_beams * vocab_size)
            next_token_scores, next_tokens = torch.topk(next_token_scores, 
                2 * num_beams, dim=1, largest=True, sorted=True)
        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        beam_outputs = beam_scorer.process(input_ids, next_token_scores,
            next_tokens, next_indices, pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        beam_scores = beam_outputs['next_beam_scores']
        beam_next_tokens = beam_outputs['next_beam_tokens']
        beam_idx = beam_outputs['next_beam_indices']
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.
            unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1
        model_kwargs = self._update_model_kwargs_for_generation(outputs,
            model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        if model_kwargs['past'] is not None:
            model_kwargs['past'] = self._reorder_cache(model_kwargs['past'],
                beam_idx)
        if model_kwargs['decoder_mention_flag'] is not None:
            model_kwargs['decoder_mention_flag'] = model_kwargs[
                'decoder_mention_flag'].index_select(0, beam_idx)
        if beam_scorer.is_done:
            break
    decoded = beam_scorer.finalize(input_ids, beam_scores, next_tokens,
        next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
    if use_constrained_decoding:
        decoded = decoded.view(batch_size, num_states, -1)
    return decoded
