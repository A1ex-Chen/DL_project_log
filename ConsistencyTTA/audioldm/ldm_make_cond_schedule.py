def make_cond_schedule(self):
    self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.
        num_timesteps - 1, dtype=torch.long)
    ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.
        num_timesteps_cond)).long()
    self.cond_ids[:self.num_timesteps_cond] = ids
