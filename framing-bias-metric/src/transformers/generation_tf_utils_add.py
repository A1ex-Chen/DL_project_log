def add(self, hyp, sum_logprobs):
    """
        Add a new hypothesis to the list.
        """
    score = sum_logprobs / len(hyp) ** self.length_penalty
    if len(self) < self.num_beams or score > self.worst_score:
        self.beams.append((score, hyp))
        if len(self) > self.num_beams:
            sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(
                self.beams)])
            del self.beams[sorted_scores[0][1]]
            self.worst_score = sorted_scores[1][0]
        else:
            self.worst_score = min(score, self.worst_score)
