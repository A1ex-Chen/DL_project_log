def test_clip_sample_range(self):
    for clip_sample_range in [1, 5, 10, 20]:
        self.check_over_configs(clip_sample_range=clip_sample_range)
