def test_clip_denoised(self):
    for clip_denoised in [True, False]:
        self.check_over_configs(clip_denoised=clip_denoised)
