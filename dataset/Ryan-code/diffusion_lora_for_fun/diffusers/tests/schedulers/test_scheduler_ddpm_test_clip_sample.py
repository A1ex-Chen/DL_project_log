def test_clip_sample(self):
    for clip_sample in [True, False]:
        self.check_over_configs(clip_sample=clip_sample)
