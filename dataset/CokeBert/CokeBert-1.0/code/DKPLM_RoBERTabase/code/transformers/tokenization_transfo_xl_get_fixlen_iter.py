def get_fixlen_iter(self, start=0):
    for i in range(start, self.data.size(0) - 1, self.bptt):
        yield self.get_batch(i)
