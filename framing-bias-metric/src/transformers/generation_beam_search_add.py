def add(self, hyp: torch.LongTensor, sum_logprobs: float):
    """
        Add a new hypothesis to the list.
        """
    score = sum_logprobs / hyp.shape[-1] ** self.length_penalty
    if len(self) < self.num_beams or score > self.worst_score:
        self.beams.append((score, hyp))
        if len(self) > self.num_beams:
            sorted_next_scores = sorted([(s, idx) for idx, (s, _) in
                enumerate(self.beams)])
            del self.beams[sorted_next_scores[0][1]]
            self.worst_score = sorted_next_scores[1][0]
        else:
            self.worst_score = min(score, self.worst_score)
