def test_clip_sample(self):
    for clip_sample_range in [1.0, 2.0, 3.0]:
        self.check_over_configs(clip_sample_range=clip_sample_range,
            clip_sample=True)
