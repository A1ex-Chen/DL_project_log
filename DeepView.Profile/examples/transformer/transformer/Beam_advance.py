def advance(self, word_prob):
    """Update beam status and check if finished or not."""
    num_words = word_prob.size(1)
    if len(self.prev_ks) > 0:
        beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
    else:
        beam_lk = word_prob[0]
    flat_beam_lk = beam_lk.view(-1)
    best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
    best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
    self.all_scores.append(self.scores)
    self.scores = best_scores
    prev_k = best_scores_id / num_words
    self.prev_ks.append(prev_k)
    self.next_ys.append(best_scores_id - prev_k * num_words)
    if self.next_ys[-1][0].item() == Constants.EOS:
        self._done = True
        self.all_scores.append(self.scores)
    return self._done
