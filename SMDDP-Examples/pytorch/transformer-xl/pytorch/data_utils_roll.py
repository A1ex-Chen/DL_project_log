def roll(self, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)
    for i in range(self.data.size(1)):
        row = self.data[:, i]
        shift = torch.randint(0, self.data.size(0), (1,), generator=rng)
        row = torch.cat((row[shift:], row[:shift]))
        self.data[:, i] = row
