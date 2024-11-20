def test_rescale_betas_zero_snr(self):
    for rescale_betas_zero_snr in [True, False]:
        self.check_over_configs(rescale_betas_zero_snr=rescale_betas_zero_snr)
