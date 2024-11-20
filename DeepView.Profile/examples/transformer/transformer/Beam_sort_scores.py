def sort_scores(self):
    """Sort the scores."""
    return torch.sort(self.scores, 0, True)
