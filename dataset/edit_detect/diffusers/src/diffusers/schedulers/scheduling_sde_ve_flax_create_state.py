def create_state(self):
    state = ScoreSdeVeSchedulerState.create()
    return self.set_sigmas(state, self.config.num_train_timesteps, self.
        config.sigma_min, self.config.sigma_max, self.config.sampling_eps)
