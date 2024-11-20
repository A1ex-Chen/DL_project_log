def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) ->torch.FloatTensor:
    stopped_samples = self._calc_stopped_samples(input_ids)
    for i, should_stop in enumerate(stopped_samples):
        if should_stop:
            scores[i, self.eos_token_id] = float(2 ** 15)
    return scores
