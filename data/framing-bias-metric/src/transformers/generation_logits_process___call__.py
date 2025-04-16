def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) ->torch.FloatTensor:
    mask = torch.full_like(scores, -math.inf)
    for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams,
        input_ids.shape[-1])):
        for beam_id, sent in enumerate(beam_sent):
            mask[batch_id * self._num_beams + beam_id, self.
                _prefix_allowed_tokens_fn(batch_id, sent)] = 0
    return scores + mask
