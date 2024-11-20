def get_fixlen_iter(self, start=0):
    if start != 0:
        start += self.bptt
    for i in range(start, self.data.size(0) - 1, self.bptt):
        self.last_iter = i
        yield self.get_batch(i)
