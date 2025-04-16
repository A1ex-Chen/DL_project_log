def check_eval_mode(self):
    super(AudioLCM, self).check_eval_mode()
    for model, name in zip([self.student_target_unet, self.ema_vae_decoder,
        self.vae.vocoder], ['student_target_unet', 'ema_vae_decoder',
        'vae.vocoder']):
        assert model.training == False, f'The {name} is not in eval mode.'
        for param in model.parameters():
            assert param.requires_grad == False, f'The {name} is not frozen.'
    for param in self.ema_vae_pqconv.parameters():
        assert param.requires_grad == False, 'The ema_vae_pqconv is not frozen.'
