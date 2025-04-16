def test_clip_sample(self):
    for clip_sample in [True, False]:
        self.check_over_configs(time_step=self.default_valid_timestep,
            clip_sample=clip_sample)
