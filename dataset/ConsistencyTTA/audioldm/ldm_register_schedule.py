def register_schedule(self, given_betas=None, beta_schedule='linear',
    timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
    super().register_schedule(given_betas, beta_schedule, timesteps,
        linear_start, linear_end, cosine_s)
    self.shorten_cond_schedule = self.num_timesteps_cond > 1
    if self.shorten_cond_schedule:
        self.make_cond_schedule()
