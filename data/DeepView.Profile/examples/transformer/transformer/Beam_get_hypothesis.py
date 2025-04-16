def get_hypothesis(self, k):
    """ Walk back to construct the full hypothesis. """
    hyp = []
    for j in range(len(self.prev_ks) - 1, -1, -1):
        hyp.append(self.next_ys[j + 1][k])
        k = self.prev_ks[j][k]
    return list(map(lambda x: x.item(), hyp[::-1]))
