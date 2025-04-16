def is_done(self, best_sum_logprobs, cur_len):
    """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
    if len(self) < self.num_beams:
        return False
    elif self.early_stopping:
        return True
    else:
        cur_score = best_sum_logprobs / cur_len ** self.length_penalty
        ret = self.worst_score >= cur_score
        return ret
