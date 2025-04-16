def get_the_best_score_and_idx(self):
    """Get the score of the best in the beam."""
    scores, ids = self.sort_scores()
    return scores[1], ids[1]
