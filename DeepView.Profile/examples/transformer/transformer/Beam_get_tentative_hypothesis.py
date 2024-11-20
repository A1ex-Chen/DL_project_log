def get_tentative_hypothesis(self):
    """Get the decoded sequence for the current timestep."""
    if len(self.next_ys) == 1:
        dec_seq = self.next_ys[0].unsqueeze(1)
    else:
        _, keys = self.sort_scores()
        hyps = [self.get_hypothesis(k) for k in keys]
        hyps = [([Constants.BOS] + h) for h in hyps]
        dec_seq = torch.LongTensor(hyps)
    return dec_seq
