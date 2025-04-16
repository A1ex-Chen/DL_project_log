def step(self, log_probabilities):
    """ Grows the beam by one step. """
    self._step += 1
    vocab_size = log_probabilities.size(-1)
    _B = log_probabilities.size(0) // self.beam_size
    log_probabilities += self.topk_log_probabilities.view(-1, 1)
    self.enforce_min_length(log_probabilities)
    if self.block_repeating_trigram:
        self.remove_repeating_trigrams(log_probabilities, _B)
    topk_log_probabilities, topk_ids = log_probabilities.topk(log_probabilities
        .view(_B, self.beam_size * vocab_size), self.beam_size, dim=1)
    topk_scores = topk_log_probabilities / self.length_penalty()
    topk_beam_ids = topk_ids.div(vocab_size)
    topk_token_ids = topk_ids.fmod(vocab_size)
    surviving_beams_rows = (topk_beam_ids + self.beam_offset[:_B].view(-1, 1)
        ).view(-1)
    self.growing_beam = torch.cat([self.growing_beam.index_select(0,
        surviving_beams_rows), topk_token_ids.view(-1, 1)], 1)
    is_finished = topk_token_ids.eq(self.end_token_id)
    self.enforce_max_length()
    is_top_beam_finished = is_finished[:, 0].eq(1)
    if is_finished.any():
        predictions = self.growing_beam.view(-1, self.beam_size, self.
            growing_beam.size(1))
        for i in range(is_finished.size(0)):
            if is_top_beam_finished[i]:
                is_finished[i].fill_(1)
            finished_hyp = is_finished[i].nonzero().view(-1)
            b = self.batch_offset[i]
            for j in finished_hyp:
                self.hypotheses[b].append((topk_scores[i, j], predictions[i,
                    j, :]))
            if is_top_beam_finished[i]:
                best_hyp = sorted(self.hypotheses[b], key=lambda x: x[0],
                    reverse=True)
                best_score, best_prediction = best_hyp[0]
                self.results['scores'][b].append(best_score)
                self.results['predictions'][b].append(best_prediction)
        non_finished = is_top_beam_finished.eq(0).nonzero().view(-1)
        if len(non_finished) == 0:
            self.is_done = True
        topk_log_probabilities = topk_log_probabilities.index_select(0,
            non_finished)
        self.batch_offset = self.batch_offset.index_select(0, non_finished)
        self.growing_beam = predictions.index_select(0, non_finished).view(
            -1, self.growing_beam.size(-1))
        surviving_beams_rows = surviving_beams_rows.index_select(0,
            non_finished)
    return surviving_beams_rows
